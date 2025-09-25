import asyncio
import logging
import json
import threading
from typing import AsyncGenerator, Optional

from PIL import Image

from trickle import media, TricklePublisher, TrickleSubscriber, InputFrame, OutputFrame, AudioFrame, AudioOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT

from .protocol import StreamProtocol
from .last_value_cache import LastValueCache

logger = logging.getLogger(__name__)


class TrickleProtocol(StreamProtocol):
    """
    Complete I/O adapter for Trickle transport with full media pipeline integration.
    Preserves all original functionality while providing clean async interface.
    """

    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: Optional[str] = None,
        events_url: Optional[str] = None,
        width: Optional[int] = DEFAULT_WIDTH,
        height: Optional[int] = DEFAULT_HEIGHT
    ):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.width = width
        self.height = height

        # Media pipeline queues (asyncio-based)
        self._ingress_queue: Optional[asyncio.Queue] = None
        self._egress_queue: Optional[asyncio.Queue] = None

        # Transport components
        self.control_subscriber: Optional[TrickleSubscriber] = None
        self.events_publisher: Optional[TricklePublisher] = None

        # Media tasks (preserve original media.run_subscribe/run_publish)
        self.subscribe_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None

        # Metadata cache for decoder/encoder coordination
        self.metadata_cache: Optional[LastValueCache] = None

        # State
        self._running = False
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start Trickle transport with complete media pipeline"""
        if self._running:
            logger.warning("TrickleProtocol already started")
            return

        logger.info(
            "Starting TrickleProtocol: sub=%s pub=%s control=%s events=%s dims=%dx%d",
            self.subscribe_url, self.publish_url, self.control_url,
            self.events_url, self.width, self.height
        )

        try:
            self._stop_event.clear()

            # Initialize queues with reasonable buffer sizes
            self._ingress_queue = asyncio.Queue(maxsize=5)
            self._egress_queue = asyncio.Queue(maxsize=5)

            # Metadata cache for decoder/encoder coordination (preserve original)
            self.metadata_cache = LastValueCache[dict]()

            # Start media tasks with original media.run_subscribe/run_publish
            self.subscribe_task = asyncio.create_task(
                media.run_subscribe(
                    self.subscribe_url,
                    self._ingress_frame_callback,  # Callback to our queue
                    self.metadata_cache.put,
                    self.emit_monitoring_event,
                    self.width,
                    self.height
                ),
                name="trickle_subscribe"
            )

            self.publish_task = asyncio.create_task(
                media.run_publish(
                    self.publish_url,
                    self._egress_frame_generator,  # Generator from our queue
                    self.metadata_cache.get,
                    self.emit_monitoring_event
                ),
                name="trickle_publish"
            )

            # Start control subscriber if configured
            if self.control_url and self.control_url.strip():
                self.control_subscriber = TrickleSubscriber(self.control_url)
                logger.info("Control subscriber started: %s", self.control_url)

            # Start events publisher if configured
            if self.events_url and self.events_url.strip():
                self.events_publisher = TricklePublisher(self.events_url, "application/json")
                logger.info("Events publisher started: %s", self.events_url)

            self._running = True
            logger.info("TrickleProtocol started successfully")

        except Exception as e:
            logger.error("Failed to start TrickleProtocol: %s", e, exc_info=True)
            await self._cleanup()
            raise

    async def stop(self):
        """Stop Trickle transport with graceful shutdown"""
        if not self._running:
            logger.debug("TrickleProtocol already stopped")
            return

        logger.info("Stopping TrickleProtocol")

        try:
            self._stop_event.set()

            # Send sentinel values to stop media tasks gracefully (preserve original)
            if self._ingress_queue:
                try:
                    self._ingress_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

            if self._egress_queue:
                try:
                    self._egress_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

            # Stop control subscriber
            if self.control_subscriber:
                await self.control_subscriber.close()
                self.control_subscriber = None

            # Stop events publisher
            if self.events_publisher:
                await self.events_publisher.close()
                self.events_publisher = None

            # Wait for media tasks to complete (preserve original timeout)
            tasks = [t for t in [self.subscribe_task, self.publish_task] if t]
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Media tasks did not stop within timeout, cancelling")
                    for task in tasks:
                        if not task.done():
                            task.cancel()

                    # Wait for cancellation
                    await asyncio.gather(*tasks, return_exceptions=True)

            await self._cleanup()

        except Exception as e:
            logger.error("Error stopping TrickleProtocol: %s", e, exc_info=True)
        finally:
            self._running = False
            logger.info("TrickleProtocol stopped")

    async def _cleanup(self):
        """Clean up resources"""
        self.subscribe_task = None
        self.publish_task = None
        self._ingress_queue = None
        self._egress_queue = None
        self.metadata_cache = None

    # StreamProtocol interface implementation with complete functionality

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        """Yield frames from media pipeline with original audio passthrough logic"""
        if not self._ingress_queue:
            logger.warning("Ingress loop called but queue not initialized")
            return

        logger.info("Ingress loop started")

        try:
            while not done.is_set():
                try:
                    # Wait for frame with timeout to check done event
                    frame = await asyncio.wait_for(
                        self._ingress_queue.get(),
                        timeout=0.5
                    )

                    if frame is None:  # Sentinel for shutdown
                        break

                    # Handle audio passthrough (preserve exact original logic)
                    if isinstance(frame, AudioFrame):
                        # TEMP: Put audio immediately into the publish queue
                        # TODO: Remove once there is ComfyUI audio support
                        if self._egress_queue:
                            try:
                                self._egress_queue.put_nowait(AudioOutput([frame]))
                            except asyncio.QueueFull:
                                logger.warning("Egress queue full, dropping audio passthrough")
                        continue

                    if isinstance(frame, InputFrame):
                        yield frame
                    else:
                        logger.warning("Unknown frame type received: %s", type(frame))

                except asyncio.TimeoutError:
                    continue  # Check done event and retry

        except Exception as e:
            logger.error("Ingress loop error: %s", e, exc_info=True)
        finally:
            logger.info("Ingress loop ended")

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        """Consume output frames and feed to media pipeline"""
        if not self._egress_queue:
            logger.warning("Egress loop called but queue not initialized")
            return

        logger.info("Egress loop started")

        try:
            async for frame in output_frames:
                if isinstance(frame, OutputFrame):
                    try:
                        await self._egress_queue.put(frame)
                    except Exception as e:
                        logger.error("Failed to enqueue output frame: %s", e)
                        break
                else:
                    logger.warning("Invalid output frame type: %s", type(frame))

        except Exception as e:
            logger.error("Egress loop error: %s", e, exc_info=True)
        finally:
            logger.info("Egress loop ended")

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        """Complete control loop with original keepalive filtering"""
        if not self.control_subscriber:
            logger.info("No control subscriber configured")
            return

        logger.info("Control loop started: %s", self.control_url)
        keepalive_message = {"keep": "alive"}  # Preserve original keepalive logic

        try:
            while not done.is_set():
                try:
                    # Get next segment with timeout to check done event
                    try:
                        segment = await asyncio.wait_for(
                            self.control_subscriber.next(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue  # Check done event and retry

                    if not segment or segment.eos():
                        logger.info("Control stream ended")
                        break

                    params_data = await segment.read()
                    if not params_data:
                        continue

                    try:
                        data = json.loads(params_data)
                        if data == keepalive_message:
                            # Ignore periodic keepalive messages (preserve original)
                            continue

                        logger.info("Received control message: %s", data)
                        yield data

                    except json.JSONDecodeError as e:
                        logger.warning("Invalid JSON in control message: %s", e)
                        continue

                except Exception as e:
                    logger.error("Control loop error: %s", e)
                    continue

        except Exception as e:
            logger.error("Control loop fatal error: %s", e, exc_info=True)
        finally:
            logger.info("Control loop ended")

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Complete event emission with original format"""
        if not self.events_publisher:
            logger.debug("No events publisher configured")
            return

        try:
            # Preserve original event envelope format
            event_json = json.dumps({
                "event": event,
                "queue_event_type": queue_event_type
            })

            async with await self.events_publisher.next() as segment:
                await segment.write(event_json.encode())

        except Exception as e:
            logger.error("Failed to emit monitoring event: %s", e)

    # Media pipeline callbacks (bridge async/sync boundaries)

    def _ingress_frame_callback(self, frame: InputFrame):
        """
        Callback for media.run_subscribe to enqueue ingress frames.
        This is called from the media pipeline thread, so we need thread-safe enqueueing.
        """
        if not self._ingress_queue or not frame:
            return

        try:
            # Use thread-safe put_nowait since this is called from media thread
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._safe_enqueue_ingress, frame)
        except Exception as e:
            logger.error("Error in ingress callback: %s", e)

    def _safe_enqueue_ingress(self, frame: InputFrame):
        """Thread-safe ingress frame enqueueing"""
        try:
            self._ingress_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Ingress queue full, dropping frame")

    def _egress_frame_generator(self):
        """
        Generator for media.run_publish to dequeue egress frames.
        This is called synchronously from the media pipeline, so we need to bridge to async.
        Preserves the exact original media.py interface.
        """
        if not self._egress_queue:
            return None

        try:
            # This is called from a thread context by media.run_publish
            # We need to get the current event loop and run the coroutine
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to use run_until_complete
                # But since we're in a thread, create a new event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._sync_get_egress_frame)
                    return future.result(timeout=30.0)  # 30s timeout
            else:
                return loop.run_until_complete(self._egress_queue.get())
        except Exception as e:
            logger.error("Error in egress generator: %s", e)
            return None

    def _sync_get_egress_frame(self):
        """Synchronous wrapper for getting egress frames"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._egress_queue.get())
            finally:
                loop.close()
        except Exception as e:
            logger.error("Error getting egress frame: %s", e)
            return None
