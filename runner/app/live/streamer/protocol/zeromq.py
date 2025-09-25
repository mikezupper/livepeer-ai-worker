import io
import time
import asyncio
import logging
from typing import AsyncGenerator, Optional
from fractions import Fraction

import zmq.asyncio
from PIL import Image
import numpy as np
import torch

from trickle import InputFrame, VideoOutput, VideoFrame
from streamer.protocol.protocol import StreamProtocol

TIME_BASE = Fraction(1, 90000)

logger = logging.getLogger(__name__)


class ZeroMQProtocol(StreamProtocol):
    """
    Complete I/O adapter for ZeroMQ transport with proper async implementation.
    Maintains exact same encoding/decoding behavior as original.
    """

    def __init__(self, input_address: str, output_address: str):
        self.input_address = input_address
        self.output_address = output_address

        # ZeroMQ context and sockets
        self.context: Optional[zmq.asyncio.Context] = None
        self.input_socket: Optional[zmq.asyncio.Socket] = None
        self.output_socket: Optional[zmq.asyncio.Socket] = None

        # State
        self._running = False

    async def start(self):
        """Start ZeroMQ sockets and connections"""
        if self._running:
            logger.warning("ZeroMQProtocol already started")
            return

        logger.info(
            "Starting ZeroMQProtocol: input=%s output=%s",
            self.input_address, self.output_address
        )

        try:
            # Create ZeroMQ context
            self.context = zmq.asyncio.Context()

            # Create and configure input socket (subscriber)
            self.input_socket = self.context.socket(zmq.SUB)
            self.input_socket.connect(self.input_address)
            self.input_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
            self.input_socket.set_hwm(10)  # High water mark for backpressure

            # Create and configure output socket (publisher)
            self.output_socket = self.context.socket(zmq.PUB)
            self.output_socket.connect(self.output_address)
            self.output_socket.set_hwm(10)  # High water mark for backpressure

            self._running = True
            logger.info("ZeroMQProtocol started successfully")

        except Exception as e:
            logger.error("Failed to start ZeroMQProtocol: %s", e, exc_info=True)
            await self._cleanup()
            raise

    async def stop(self):
        """Stop ZeroMQ sockets and cleanup"""
        if not self._running:
            logger.debug("ZeroMQProtocol already stopped")
            return

        logger.info("Stopping ZeroMQProtocol")

        try:
            await self._cleanup()
        except Exception as e:
            logger.error("Error stopping ZeroMQProtocol: %s", e, exc_info=True)
        finally:
            self._running = False
            logger.info("ZeroMQProtocol stopped")

    async def _cleanup(self):
        """Clean up ZeroMQ resources"""
        # Close sockets
        if self.input_socket:
            self.input_socket.close()
            self.input_socket = None

        if self.output_socket:
            self.output_socket.close()
            self.output_socket = None

        # Terminate context
        if self.context:
            self.context.term()
            self.context = None

    # StreamProtocol interface implementation

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        """Yield decoded video frames from ZeroMQ input socket"""
        if not self.input_socket:
            logger.warning("Ingress loop called but input socket not initialized")
            return

        logger.info("ZeroMQ ingress loop started")

        try:
            while not done.is_set():
                try:
                    # Receive frame with timeout
                    frame_bytes = await asyncio.wait_for(
                        self.input_socket.recv(),
                        timeout=1.0
                    )

                    # Decode JPEG bytes to tensor (preserve original logic exactly)
                    try:
                        tensor = from_jpeg_bytes(frame_bytes)
                        timestamp = int(time.time() / TIME_BASE)
                        frame = InputFrame.from_av_video(tensor, timestamp, TIME_BASE)
                        yield frame

                    except Exception as e:
                        logger.error("Failed to decode frame: %s", e)
                        continue

                except asyncio.TimeoutError:
                    continue  # Check done event and retry
                except zmq.Again:
                    # No message available, short wait
                    await asyncio.sleep(0.01)
                    continue

        except Exception as e:
            logger.error("ZeroMQ ingress loop error: %s", e, exc_info=True)
        finally:
            logger.info("ZeroMQ ingress loop ended")

    async def egress_loop(self, output_frames: AsyncGenerator[VideoOutput, None]):
        """Consume output frames and encode to ZeroMQ output socket"""
        if not self.output_socket:
            logger.warning("Egress loop called but output socket not initialized")
            return

        logger.info("ZeroMQ egress loop started")

        try:
            async for output in output_frames:
                if isinstance(output, VideoOutput):
                    try:
                        # Encode tensor to JPEG bytes (preserve original logic)
                        frame_bytes = to_jpeg_bytes(output.tensor)

                        # Send to ZeroMQ socket (non-blocking)
                        await self.output_socket.send(frame_bytes, flags=zmq.NOBLOCK)

                        logger.debug("Sent frame: size=%d bytes", len(frame_bytes))

                    except zmq.Again:
                        logger.warning("Output socket busy, dropping frame")
                        continue
                    except Exception as e:
                        logger.error("Failed to send frame: %s", e)
                        continue
                else:
                    logger.warning("Invalid output type for ZeroMQ: %s", type(output))

        except Exception as e:
            logger.error("ZeroMQ egress loop error: %s", e, exc_info=True)
        finally:
            logger.info("ZeroMQ egress loop ended")

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        """No-op control loop - ZeroMQ doesn't support control messages"""
        logger.info("ZeroMQ control loop: no control channel support, waiting for done")

        # Just wait for done event since ZeroMQ doesn't have control channel
        try:
            await done.wait()
        except Exception:
            pass

        # Empty generator with proper typing (preserve original behavior)
        if False:
            yield {}

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """No-op event emission - ZeroMQ doesn't support events"""
        logger.debug("ZeroMQ emit_monitoring_event: no events channel support (event ignored)")


# Utility functions for JPEG encoding/decoding (preserve exact original logic)

def to_jpeg_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor to JPEG bytes - exact original implementation"""
    try:
        # Convert tensor to numpy array (preserve original conversion)
        image_np = (tensor * 255).byte().cpu().numpy()
        image = Image.fromarray(image_np)

        # Encode to JPEG with original settings
        buffer = io.BytesIO()
        try:
            image.save(buffer, format="JPEG", quality=85, optimize=True)
            return buffer.getvalue()
        finally:
            buffer.close()

    except Exception as e:
        logger.error("Failed to encode tensor to JPEG: %s", e)
        raise


def from_jpeg_bytes(frame_bytes: bytes) -> torch.Tensor:
    """Convert JPEG bytes to tensor - exact original implementation"""
    try:
        # Decode JPEG to PIL Image
        image = Image.open(io.BytesIO(frame_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        target_size = 512  # Preserve original fixed size

        # Crop to center square if not already square (preserve original logic)
        if (width, height) != (target_size, target_size):
            square_size = target_size
            start_x = width // 2 - square_size // 2
            start_y = height // 2 - square_size // 2
            image = image.crop((
                start_x, start_y,
                start_x + square_size,
                start_y + square_size
            ))

        # Convert to tensor (preserve exact original format)
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.tensor(image_np).unsqueeze(0)

    except Exception as e:
        logger.error("Failed to decode JPEG bytes: %s", e)
        raise
