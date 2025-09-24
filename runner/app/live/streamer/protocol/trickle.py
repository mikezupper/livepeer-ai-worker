import asyncio
import logging
import json
from typing import AsyncGenerator, Optional

from .protocol import StreamProtocol
from .last_value_cache import LastValueCache
from app.live.trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT, InputFrame, OutputFrame, TrickleSubscriber, TricklePublisher, \
    AudioFrame, AudioOutput, media


class TrickleProtocol(StreamProtocol):
    """
    Trickle protocol adapter that provides pure I/O interface.

    Responsibilities:
    - Expose async generators for ingress/egress/control
    - Handle Trickle wire protocol (subscribe/publish)
    - Emit monitoring events
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

        # Internal transport state
        self._ingress_queue: Optional[asyncio.Queue[InputFrame]] = None
        self._egress_queue: Optional[asyncio.Queue[OutputFrame]] = None
        self._metadata_cache: Optional[LastValueCache] = None

        # Transport tasks (managed internally)
        self._subscribe_task: Optional[asyncio.Task] = None
        self._publish_task: Optional[asyncio.Task] = None

        # Control and events clients
        self._control_subscriber: Optional[TrickleSubscriber] = None
        self._events_publisher: Optional[TricklePublisher] = None

        # Shutdown coordination
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the transport layer."""
        if self._ingress_queue is not None:
            raise RuntimeError("Protocol already started")

        logging.info(f"Starting TrickleProtocol: subscribe={self.subscribe_url} publish={self.publish_url}")

        # Create internal queues for transport boundary
        self._ingress_queue = asyncio.Queue(maxsize=10)  # Small buffer for backpressure
        self._egress_queue = asyncio.Queue(maxsize=10)
        self._metadata_cache = LastValueCache()

        # Reset stop event
        self._stop_event.clear()

        # Start media transport tasks
        self._subscribe_task = asyncio.create_task(
            self._run_subscribe_adapter()
        )
        self._publish_task = asyncio.create_task(
            self._run_publish_adapter()
        )

        # Initialize control and events channels
        if self.control_url and self.control_url.strip():
            self._control_subscriber = TrickleSubscriber(self.control_url)

        if self.events_url and self.events_url.strip():
            self._events_publisher = TricklePublisher(self.events_url, "application/json")

    async def stop(self):
        """Stop the transport layer."""
        if self._ingress_queue is None:
            return  # Already stopped

        logging.info("Stopping TrickleProtocol")

        # Signal stop to transport tasks
        self._stop_event.set()

        # Send sentinel values to queues to unblock tasks
        try:
            await self._ingress_queue.put(None)
        except:
            pass

        try:
            await self._egress_queue.put(None)
        except:
            pass

        # Wait for transport tasks to complete
        tasks = []
        if self._subscribe_task:
            tasks.append(self._subscribe_task)
        if self._publish_task:
            tasks.append(self._publish_task)

        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                logging.warning("Transport tasks did not stop within timeout, cancelling")
                for task in tasks:
                    task.cancel()

        # Clean up transport clients
        if self._control_subscriber:
            await self._control_subscriber.close()
            self._control_subscriber = None

        if self._events_publisher:
            await self._events_publisher.close()
            self._events_publisher = None

        # Reset state
        self._subscribe_task = None
        self._publish_task = None
        self._ingress_queue = None
        self._egress_queue = None
        self._metadata_cache = None

        logging.info("TrickleProtocol stopped")

    async def _run_subscribe_adapter(self):
        """Adapter task that runs media.run_subscribe and feeds our ingress queue."""
        try:
            def frame_callback(frame: InputFrame):
                if frame is None:
                    return
                # Non-blocking put to queue - if queue is full, we drop frames for backpressure
                try:
                    self._ingress_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    logging.warning("Ingress queue full, dropping frame")

            def metadata_callback(metadata: dict):
                if self._metadata_cache:
                    self._metadata_cache.put(metadata)

            async def monitoring_callback(event_data: dict, queue_event_type: str = "stream_trace"):
                await self.emit_monitoring_event(event_data, queue_event_type)

            # Use the existing media.run_subscribe infrastructure
            await media.run_subscribe(
                self.subscribe_url,
                frame_callback,
                metadata_callback,
                monitoring_callback,
                self.width,
                self.height
            )

        except Exception as e:
            logging.error(f"Error in subscribe adapter: {e}", exc_info=True)
        finally:
            # Signal end of stream
            try:
                await self._ingress_queue.put(None)
            except:
                pass

    async def _run_publish_adapter(self):
        """Adapter task that consumes our egress queue and runs media.run_publish."""
        try:
            def frame_generator():
                """Generator that bridges asyncio.Queue to sync interface."""
                while True:
                    # This will be called from the media thread
                    try:
                        # We need to bridge async queue to sync - use a simple approach
                        future = asyncio.run_coroutine_threadsafe(
                            self._egress_queue.get(),
                            asyncio.get_event_loop()
                        )
                        frame = future.result(timeout=1.0)
                        if frame is None:
                            break
                        return frame
                    except Exception as e:
                        logging.error(f"Error getting frame from egress queue: {e}")
                        break
                return None

            def metadata_getter():
                """Get metadata from cache."""
                if self._metadata_cache:
                    return self._metadata_cache.get()
                return None

            async def monitoring_callback(event_data: dict, queue_event_type: str = "stream_trace"):
                await self.emit_monitoring_event(event_data, queue_event_type)

            # Use the existing media.run_publish infrastructure
            await media.run_publish(
                self.publish_url,
                frame_generator,
                metadata_getter,
                monitoring_callback
            )

        except Exception as e:
            logging.error(f"Error in publish adapter: {e}", exc_info=True)

    async def ingress_loop(self) -> AsyncGenerator[InputFrame, None]:
        """Generator that yields frames from the ingress transport."""
        if self._ingress_queue is None:
            raise RuntimeError("Protocol not started")

        while not self._stop_event.is_set():
            try:
                # Get frame from transport queue with timeout
                frame = await asyncio.wait_for(self._ingress_queue.get(), timeout=1.0)
                if frame is None:
                    break  # End of stream sentinel

                # Handle audio passthrough (preserve existing behavior)
                if isinstance(frame, AudioFrame):
                    # Put audio directly to egress queue for passthrough
                    try:
                        await self._egress_queue.put(AudioOutput([frame]))
                    except:
                        pass  # Queue might be closed

                yield frame

            except asyncio.TimeoutError:
                # Timeout is normal - just check stop condition and continue
                continue
            except Exception as e:
                logging.error(f"Error in ingress loop: {e}")
                break

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        """Consume output frames and send to transport."""
        if self._egress_queue is None:
            raise RuntimeError("Protocol not started")

        try:
            async for frame in output_frames:
                if self._stop_event.is_set():
                    break

                # Send frame to transport queue
                try:
                    await self._egress_queue.put(frame)
                except Exception as e:
                    logging.error(f"Error putting frame to egress queue: {e}")
                    break

        except Exception as e:
            logging.error(f"Error in egress loop: {e}")
        finally:
            # Send sentinel to stop publish adapter
            try:
                await self._egress_queue.put(None)
            except:
                pass

    async def control_loop(self) -> AsyncGenerator[dict, None]:
        """Generator that yields control parameter updates."""
        if not self._control_subscriber:
            logging.warning("No control-url provided, inference won't get updates from control trickle subscription")
            return

        logging.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        while not self._stop_event.is_set():
            try:
                segment = await self._control_subscriber.next()
                if not segment or segment.eos():
                    break

                params = await segment.read()
                data = json.loads(params)
                if data == keepalive_message:
                    continue  # Ignore keepalive messages

                logging.info("Received control message with params: %s", data)
                yield data

            except Exception as e:
                logging.error(f"Error in control loop", exc_info=True)
                # Don't break on individual errors - keep trying
                await asyncio.sleep(1.0)

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Emit a monitoring event via the events channel."""
        if not self._events_publisher:
            return

        try:
            event_json = json.dumps({"event": event, "queue_event_type": queue_event_type})
            async with await self._events_publisher.next() as event_segment:
                await event_segment.write(event_json.encode())
        except Exception as e:
            logging.error(f"Error emitting monitoring event: {e}")
