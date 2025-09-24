import asyncio
import logging

logger = logging.getLogger("pipeline_streamer")


class PipelineStreamer:
    """
    Single responsibility: route frames protocol.recv -> guardian.handle -> protocol.send
    - One task for the routing loop
    - No mixed lifecycle concerns
    """

    def __init__(self, protocol, process):
        self.protocol = protocol
        self.process = process
        self._task: asyncio.Task | None = None
        self._stopping = False

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run_loop(), name="pipeline_streamer_loop")
        logger.info("PipelineStreamer started")

    async def stop(self):
        self._stopping = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                logger.warning("PipelineStreamer stop timed out; cancelling loop")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        self._task = None
        logger.info("PipelineStreamer stopped")

    async def _run_loop(self):
        logger.info("PipelineStreamer loop starting")
        while not self._stopping:
            frame = await self.protocol.recv()
            if frame is None:
                # End-of-stream or no frame available; yield briefly
                await asyncio.sleep(0.005)
                continue

            try:
                processed = await self.process.handle(frame)
            except Exception as e:
                logger.error("Processing error: %s", e)
                # Optionally publish an event via protocol
                continue

            try:
                await self.protocol.send(processed)
            except Exception as e:
                logger.error("Send error: %s", e)
                continue

        logger.info("PipelineStreamer loop exiting")
