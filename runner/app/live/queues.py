import asyncio
import time
import logging

logger = logging.getLogger("queues")


class FrameQueue(asyncio.Queue):
    """
    Optional audited frame queue with timestamps for backpressure/diagnostics.
    Not required by the refactor, but ready for future needs.
    """

    async def put_frame(self, frame):
        item = {"frame": frame, "ts": time.time()}
        await self.put(item)
        logger.debug("Enqueued frame; size=%d", self.qsize())

    async def get_frame(self):
        item = await self.get()
        logger.debug("Dequeued frame; size=%d", self.qsize())
        return item["frame"]
