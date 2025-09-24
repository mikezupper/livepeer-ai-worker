import asyncio
import logging
import contextlib

logger = logging.getLogger("trickle_protocol")


class TrickleProtocol:
    """
    Transport-only responsibilities:
    - start(): establish connections, launch rx loop
    - stop(): cancel rx loop, close connections
    - recv(): await next decoded frame from internal rx queue
    - send(frame): encode & publish frame
    Control/events can be implemented internally as needed (kept minimal here).
    """

    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: str | None,
        events_url: str | None,
        request_id: str,
        stream_id: str,
    ):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.request_id = request_id
        self.stream_id = stream_id

        self._stop_event = asyncio.Event()
        self._rx_task: asyncio.Task | None = None
        self._rx_queue: asyncio.Queue = asyncio.Queue()

        # Placeholder connection handles
        self._rx_conn = None
        self._tx_conn = None

    async def start(self):
        logger.info(
            "Trickle start sub=%s pub=%s req=%s stream=%s",
            self.subscribe_url, self.publish_url, self.request_id, self.stream_id
        )
        self._stop_event.clear()
        # TODO: open connections (_rx_conn, _tx_conn)
        self._rx_task = asyncio.create_task(self._rx_loop(), name="trickle_rx_loop")

    async def stop(self):
        logger.info("Trickle stop req=%s stream=%s", self.request_id, self.stream_id)
        self._stop_event.set()
        if self._rx_task:
            self._rx_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rx_task
        self._rx_task = None
        # TODO: close connections
        self._rx_conn = None
        self._tx_conn = None

    async def recv(self):
        try:
            frame = await asyncio.wait_for(self._rx_queue.get(), timeout=0.5)
            return frame
        except asyncio.TimeoutError:
            if self._stop_event.is_set():
                return None
            return None

    async def send(self, frame):
        # TODO: serialize & publish frame via _tx_conn
        logger.debug("Trickle send frame req=%s stream=%s", self.request_id, self.stream_id)

    async def _rx_loop(self):
        logger.info("Trickle rx loop started req=%s stream=%s", self.request_id, self.stream_id)
        while not self._stop_event.is_set():
            # TODO: read trickle message, decode -> frame
            await asyncio.sleep(0.01)  # simulate I/O
            frame = {"stream_id": self.stream_id, "payload": b"", "meta": {"req": self.request_id}}
            await self._rx_queue.put(frame)
        logger.info("Trickle rx loop exiting req=%s stream=%s", self.request_id, self.stream_id)
