import asyncio
import logging
import signal
from enum import Enum

logger = logging.getLogger("lifecycle")


class LifecycleState(Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class LifecycleController:
    """
    Single lifecycle authority ensuring deterministic start/stop:
    - Start order: protocol -> guardian -> streamer
    - Stop order: streamer -> guardian -> protocol
    - Idempotent, explicit state transitions
    """

    def __init__(self, components: list):
        self.components = components
        self._stop_event = asyncio.Event()
        self._state = LifecycleState.INIT
        self._started = False

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self):
        if self._started:
            return
        logger.info("Lifecycle start (%d components)", len(self.components))
        self._state = LifecycleState.RUNNING
        for c in self.components:
            if hasattr(c, "start"):
                await c.start()
        self._started = True
        logger.info("Lifecycle state=%s", self._state.value)

    async def stop(self):
        if self._state in (LifecycleState.STOPPING, LifecycleState.STOPPED):
            self._stop_event.set()
            return
        logger.info("Lifecycle stop requested")
        self._state = LifecycleState.STOPPING
        for c in reversed(self.components):
            if hasattr(c, "stop"):
                try:
                    await c.stop()
                except Exception as e:
                    logger.error("Error stopping %s: %s", type(c).__name__, e)
        self._state = LifecycleState.STOPPED
        self._stop_event.set()
        self._started = False
        logger.info("Lifecycle state=%s", self._state.value)

    async def run_until_signal(self):
        loop = asyncio.get_running_loop()

        def _on_signal(sig):
            logger.info("Received signal: %s", sig.name)
            asyncio.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: _on_signal(s))
            except NotImplementedError:
                logger.warning("Signal handlers not supported; rely on explicit stop")

        await self.start()
        await self._stop_event.wait()
        logger.info("Lifecycle finished (state=%s)", self._state.value)
