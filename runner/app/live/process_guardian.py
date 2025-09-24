import asyncio
import logging
import contextlib

logger = logging.getLogger("process_guardian")


class ProcessGuardian:
    """
    Supervises the pipeline runtime and provides a simple per-frame handle().
    No transport routing or lifecycle mixing.
    """

    def __init__(self, pipeline: str, params: dict):
        self.pipeline = pipeline
        self.params = params or {}
        self._proc = None
        self._started = False

    async def start(self):
        if self._started:
            return
        logger.info("Guardian starting pipeline='%s' params=%s", self.pipeline, self.params)
        # Replace with actual boot logic:
        # self._proc = await asyncio.create_subprocess_exec(...)
        await asyncio.sleep(0.01)  # simulate init
        self._started = True
        logger.info("Guardian started pipeline='%s'", self.pipeline)

    async def stop(self):
        if not self._started:
            return
        logger.info("Guardian stopping pipeline='%s'", self.pipeline)
        if self._proc:
            self._proc.terminate()
            with contextlib.suppress(ProcessLookupError):
                await self._proc.wait()
            self._proc = None
        self._started = False
        logger.info("Guardian stopped pipeline='%s'", self.pipeline)

    async def handle(self, frame):
        """
        Perform inference on a single frame.
        Replace with actual IPC/RPC/shared memory interaction.
        """
        if not self._started:
            await self.start()
        # TODO: Marshal 'frame' to pipeline and return processed result
        return frame  # placeholder passthrough
