import asyncio
import logging
import time
from typing import AsyncGenerator, Optional
from enum import Enum

from .process_manager import ProcessManager
from .protocol.protocol import StreamProtocol
from .status import timestamp_to_ms
from app.live.trickle import InputFrame, OutputFrame, AudioFrame, VideoFrame, AudioOutput, VideoOutput


class StreamSessionState(Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    ERROR = "ERROR"


class StreamSession:
    """
    Single orchestrator that owns a streaming session lifecycle.

    Replaces PipelineStreamer + its supervisor + parts of ProcessGuardian orchestration.
    Owns transport protocol, task coordination, and policy decisions.
    """

    def __init__(
        self,
        protocol: StreamProtocol,
        process_manager: ProcessManager,
        request_id: str,
        manifest_id: str,
        stream_id: str,
    ):
        self.protocol = protocol
        self.process_manager = process_manager
        self.request_id = request_id
        self.manifest_id = manifest_id
        self.stream_id = stream_id

        self.state = StreamSessionState.STOPPED
        self.error_reason: Optional[str] = None

        self._task_group: Optional[asyncio.TaskGroup] = None
        self._emit_event_lock = asyncio.Lock()

        # Health monitoring
        self._last_health_check = time.time()

    async def start(self, params: dict):
        """Start the streaming session with given parameters."""
        if self.state != StreamSessionState.STOPPED:
            raise RuntimeError(f"Cannot start session in state {self.state}")

        logging.info(f"Starting stream session: request_id={self.request_id}")
        self.state = StreamSessionState.STARTING
        self.error_reason = None

        try:
            # Reset process manager for this stream session
            await self.process_manager.reset_stream(
                self.request_id,
                self.manifest_id,
                self.stream_id,
                params
            )

            # Start the transport protocol
            await self.protocol.start()

            # Emit the stream request received trace event
            await self._emit_trace_event({
                "type": "runner_receive_stream_request",
                "timestamp": int(time.time() * 1000),
            })

            # Start all tasks in a single TaskGroup
            self.state = StreamSessionState.RUNNING
            await self._start_task_group()

        except Exception as e:
            self.state = StreamSessionState.ERROR
            self.error_reason = str(e)
            logging.error(f"Error starting stream session: {e}", exc_info=True)
            # Clean up on failed start
            try:
                await self.protocol.stop()
            except Exception:
                logging.error("Error stopping protocol during failed start", exc_info=True)
            raise

    async def _start_task_group(self):
        """Start the main TaskGroup with all session tasks."""
        async with asyncio.TaskGroup() as tg:
            self._task_group = tg

            # Five main tasks that coordinate the session
            tg.create_task(self._ingress_task(), name="ingress")
            tg.create_task(self._egress_task(), name="egress")
            tg.create_task(self._control_task(), name="control")
            tg.create_task(self._status_task(), name="status")
            tg.create_task(self._health_task(), name="health")

            # TaskGroup will wait for all tasks or cancel all on first exception

        # If we reach here, all tasks completed normally (shouldn't happen in normal operation)
        if self.state == StreamSessionState.RUNNING:
            self.state = StreamSessionState.STOPPED
        self._task_group = None

    async def _ingress_task(self):
        """Pump frames from protocol to process manager."""
        try:
            async for frame in self.protocol.ingress_loop():
                if isinstance(frame, AudioFrame):
                    # Pass audio directly to process manager
                    self.process_manager.send_input(frame)
                elif isinstance(frame, VideoFrame):
                    # Pass video frame to process manager
                    self.process_manager.send_input(frame)
                else:
                    logging.warning(f"Unknown frame type received in ingress: {type(frame)}")

        except Exception as e:
            logging.error(f"Error in ingress task: {e}", exc_info=True)
            await self._handle_task_error("ingress", e)

    async def _egress_task(self):
        """Pump processed frames from process manager to protocol."""
        try:
            async def output_generator() -> AsyncGenerator[OutputFrame, None]:
                while self.state == StreamSessionState.RUNNING:
                    output = await self.process_manager.recv_output()
                    if not output:
                        continue

                    # Filter by request ID to avoid cross-session contamination
                    if isinstance(output, (VideoOutput, AudioOutput)):
                        if output.request_id != self.request_id:
                            logging.warning(
                                f"Output request ID mismatch: expected {self.request_id}, "
                                f"got {output.request_id}, dropping frame"
                            )
                            continue

                    yield output

            await self.protocol.egress_loop(output_generator())

        except Exception as e:
            logging.error(f"Error in egress task: {e}", exc_info=True)
            await self._handle_task_error("egress", e)

    async def _control_task(self):
        """Process control messages from protocol."""
        try:
            async for params in self.protocol.control_loop():
                try:
                    await self.process_manager.update_params(params)
                except Exception as e:
                    logging.error(f"Error updating params from control: {e}", exc_info=True)
                    # Don't fail the whole session for param update errors

        except Exception as e:
            logging.error(f"Error in control task: {e}", exc_info=True)
            await self._handle_task_error("control", e)

    async def _status_task(self):
        """Periodically report status via events."""
        try:
            STATUS_INTERVAL = 10.0  # seconds

            while self.state == StreamSessionState.RUNNING:
                await asyncio.sleep(STATUS_INTERVAL)

                # Get status and emit event
                status = self.process_manager.get_status(clear_transient=True)
                await self._emit_monitoring_event(status.model_dump())

        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            pass
        except Exception as e:
            logging.error(f"Error in status task: {e}", exc_info=True)
            await self._handle_task_error("status", e)

    async def _health_task(self):
        """Monitor health and implement session policies."""
        try:
            HEALTH_CHECK_INTERVAL = 1.0  # seconds
            IDLE_SOFT_STOP_TIMEOUT = 60.0  # seconds
            IDLE_ERROR_TIMEOUT = 90.0  # seconds

            while self.state == StreamSessionState.RUNNING:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

                current_time = time.time()
                status = self.process_manager.get_status()

                # Check for process errors
                last_error = self.process_manager.get_last_error()
                if last_error:
                    error_msg, error_time = last_error
                    await self._emit_monitoring_event({
                        "type": "error",
                        "pipeline": self.process_manager.pipeline,
                        "message": error_msg,
                        "time": error_time,
                    })

                # Implement idleness policy
                last_input_time = status.input_status.last_input_time
                if last_input_time:
                    time_since_input = current_time - (last_input_time / 1000.0)  # status uses ms

                    if time_since_input > IDLE_ERROR_TIMEOUT:
                        # Too idle - mark as error to trigger container restart
                        await self._handle_session_error(
                            f"Stream idle for {time_since_input:.1f}s, marking as error"
                        )
                        break
                    elif time_since_input > IDLE_SOFT_STOP_TIMEOUT:
                        # Idle but not critical - trigger graceful stop
                        logging.info(f"Stream idle for {time_since_input:.1f}s, triggering graceful stop")
                        await self.stop()
                        break

                # Check if process is still healthy
                if not self.process_manager.is_alive():
                    await self._handle_session_error("Process manager is no longer alive")
                    break

        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            pass
        except Exception as e:
            logging.error(f"Error in health task: {e}", exc_info=True)
            await self._handle_task_error("health", e)

    async def _handle_task_error(self, task_name: str, error: Exception):
        """Handle errors from individual tasks."""
        error_msg = f"Task {task_name} failed: {error}"
        await self._handle_session_error(error_msg)

    async def _handle_session_error(self, reason: str):
        """Handle session-level errors."""
        logging.error(f"Session error: {reason}")
        self.state = StreamSessionState.ERROR
        self.error_reason = reason

        # Cancel the task group to stop all tasks
        if self._task_group:
            # The task group will handle cancellation
            pass

    async def stop(self, timeout: float = 10.0):
        """Stop the streaming session."""
        if self.state in (StreamSessionState.STOPPED, StreamSessionState.DRAINING):
            return

        logging.info(f"Stopping stream session: request_id={self.request_id}")
        prev_state = self.state
        self.state = StreamSessionState.DRAINING

        try:
            # If tasks are running, we need to let them complete or timeout
            if self._task_group:
                # Tasks will see state change and exit naturally
                # TaskGroup cleanup will happen in _start_task_group
                try:
                    await asyncio.wait_for(self._wait_for_tasks(), timeout=timeout)
                except asyncio.TimeoutError:
                    logging.warning(f"Tasks did not stop within {timeout}s timeout")

            # Stop the protocol
            await self.protocol.stop()

        except Exception as e:
            logging.error(f"Error during session stop: {e}", exc_info=True)
        finally:
            self.state = StreamSessionState.STOPPED
            logging.info(f"Stream session stopped: request_id={self.request_id}")

    async def _wait_for_tasks(self):
        """Wait for the task group to complete."""
        # The _start_task_group method will complete when all tasks are done
        # This is a bit of a hack since TaskGroup doesn't expose a wait method
        while self._task_group is not None:
            await asyncio.sleep(0.1)

    async def update_params(self, params: dict):
        """Update parameters for the session."""
        if self.state != StreamSessionState.RUNNING:
            raise RuntimeError(f"Cannot update params in state {self.state}")

        await self.process_manager.update_params(params)

    def is_running(self) -> bool:
        """Check if session is currently running."""
        return self.state == StreamSessionState.RUNNING

    def get_state(self) -> StreamSessionState:
        """Get current session state."""
        return self.state

    def get_error_reason(self) -> Optional[str]:
        """Get error reason if in error state."""
        return self.error_reason

    async def wait(self, timeout: float = 0) -> bool:
        """Wait for the session to complete."""
        if self.state == StreamSessionState.STOPPED:
            return True

        if timeout > 0:
            try:
                await asyncio.wait_for(self._wait_for_tasks(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False
        else:
            await self._wait_for_tasks()
            return True

    async def _emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Emit a monitoring event via the protocol."""
        event["timestamp"] = timestamp_to_ms(time.time())
        logging.info(f"Emitting monitoring event: {event}")

        async with self._emit_event_lock:
            try:
                await self.protocol.emit_monitoring_event(event, queue_event_type)
            except Exception as e:
                logging.error(f"Failed to emit monitoring event: {e}")

    async def _emit_trace_event(self, event: dict):
        """Emit a trace event."""
        await self._emit_monitoring_event(event, queue_event_type="stream_trace")
