import asyncio
import logging
import time
from typing import Optional, Dict, Any
from enum import Enum

from process_manager import ProcessManager
from streamer.protocol.protocol import StreamProtocol
from streamer.status import PipelineState, PipelineStatus, timestamp_to_ms


logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle states"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    ERROR = "ERROR"


class StreamSession:
    """
    Complete orchestrator for streaming sessions with all original functionality preserved.
    Replaces PipelineStreamer + ProcessGuardian coordination with single clear owner.
    """

    def __init__(
        self,
        pipeline: str,
        protocol: StreamProtocol,
        request_id: str,
        manifest_id: str,
        stream_id: str,
    ):
        self.pipeline = pipeline
        self.protocol = protocol
        self.request_id = request_id
        self.manifest_id = manifest_id
        self.stream_id = stream_id

        # Components
        self.process_manager: Optional[ProcessManager] = None

        # State management
        self.state = SessionState.STOPPED
        self.state_reason: Optional[str] = None
        self.start_time: Optional[float] = None

        # Task supervision
        self.task_group: Optional[asyncio.TaskGroup] = None
        self._stop_event = asyncio.Event()
        self._running = False
        self._tasks = []

        # Health monitoring (preserve original timeouts)
        self._last_health_check = time.time()

    async def start(self, params: Dict[str, Any]) -> None:
        """Start streaming session with complete original functionality"""
        if self._running:
            logger.info("Session already running, ignoring start request")
            return

        try:
            self.state = SessionState.STARTING
            self.start_time = time.time()
            self._stop_event.clear()

            # Initialize ProcessManager (subprocess)
            if not self.process_manager:
                self.process_manager = ProcessManager(self.pipeline, params)

            await self.process_manager.start()
            await self.process_manager.reset_stream(
                self.request_id,
                self.manifest_id,
                self.stream_id,
                params
            )

            # Start protocol (transport)
            await self.protocol.start()

            # Start all tasks with proper supervision
            await self._start_tasks()

            self.state = SessionState.RUNNING
            self._running = True

            logger.info(
                "StreamSession started: pipeline=%s request_id=%s stream_id=%s",
                self.pipeline, self.request_id, self.stream_id
            )

        except Exception as e:
            self.state = SessionState.ERROR
            self.state_reason = f"Start failed: {e}"
            logger.error("Failed to start StreamSession: %s", e, exc_info=True)
            await self._cleanup()
            raise

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop session with graceful shutdown"""
        if not self._running:
            return

        try:
            logger.info("Stopping StreamSession: request_id=%s", self.request_id)
            self.state = SessionState.DRAINING
            self._stop_event.set()

            # Cancel all tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Task shutdown timed out")

            await self._cleanup()

            self.state = SessionState.STOPPED
            self._running = False

            logger.info("StreamSession stopped: request_id=%s", self.request_id)

        except Exception as e:
            self.state = SessionState.ERROR
            self.state_reason = f"Stop failed: {e}"
            logger.error("Error stopping StreamSession: %s", e, exc_info=True)
            await self._cleanup()
            self._running = False

    async def update_params(self, params: Dict[str, Any]) -> None:
        """Forward parameter updates to ProcessManager"""
        if not self.process_manager:
            raise RuntimeError("Session not started")
        await self.process_manager.update_params(params)

    def is_running(self) -> bool:
        """Check if session is running"""
        return self._running and self.state == SessionState.RUNNING

    async def wait(self, timeout: float = 0) -> None:
        """Wait for session to complete"""
        if not self._tasks:
            return

        try:
            if timeout > 0:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout
                )
            else:
                await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning("Session wait timed out")

    # Implementation of StreamerCallbacks interface for ProcessManager compatibility
    def is_stream_running(self) -> bool:
        """Callback for ProcessManager compatibility"""
        return self.is_running()

    async def emit_monitoring_event(self, event_data: dict) -> None:
        """Callback for ProcessManager compatibility"""
        await self.protocol.emit_monitoring_event(event_data)

    def trigger_stop_stream(self) -> bool:
        """Callback for ProcessManager compatibility"""
        if not self._stop_event.is_set():
            self._stop_event.set()
            return True
        return False

    async def _start_tasks(self) -> None:
        """Start all session tasks with proper error handling"""
        self._tasks = []

        # Create tasks with proper exception handling
        self._tasks.extend([
            self._create_task(self._ingress_task(), "ingress"),
            self._create_task(self._egress_task(), "egress"),
            self._create_task(self._control_task(), "control"),
            self._create_task(self._status_task(), "status"),
            self._create_task(self._health_task(), "health"),
        ])

    def _create_task(self, coro, name: str):
        """Create task with proper error handling"""
        async def wrapped_task():
            try:
                await coro
            except asyncio.CancelledError:
                logger.info("Task %s cancelled", name)
                raise
            except Exception as e:
                logger.error("Task %s failed: %s", name, e, exc_info=True)
                # Signal session to stop on task failure
                self._stop_event.set()
                raise

        return asyncio.create_task(wrapped_task(), name=f"session_{name}")

    async def _cleanup(self) -> None:
        """Clean up all resources"""
        cleanup_errors = []

        # Stop protocol
        if self.protocol:
            try:
                await self.protocol.stop()
            except Exception as e:
                cleanup_errors.append(f"protocol: {e}")

        # Stop process manager
        if self.process_manager:
            try:
                await self.process_manager.stop()
            except Exception as e:
                cleanup_errors.append(f"process: {e}")

        self._tasks = []

        if cleanup_errors:
            logger.warning("Cleanup errors: %s", "; ".join(cleanup_errors))

    # Task implementations with complete original functionality

    async def _ingress_task(self) -> None:
        """Route frames from protocol to process manager (preserves original logic)"""
        logger.info("Ingress task started")

        try:
            async for frame in self.protocol.ingress_loop(self._stop_event):
                if self._stop_event.is_set():
                    break

                try:
                    self.process_manager.send_input(frame)
                except Exception as e:
                    logger.error("Failed to send input frame: %s", e)

        except Exception as e:
            logger.error("Ingress task error: %s", e, exc_info=True)
            raise
        finally:
            logger.info("Ingress task ended")

    async def _egress_task(self) -> None:
        """Route frames from process manager to protocol (preserves request_id filtering)"""
        logger.info("Egress task started")

        try:
            async def output_generator():
                while not self._stop_event.is_set():
                    try:
                        output = await self.process_manager.recv_output()
                        if not output:
                            continue

                        # Preserve original request_id filtering logic
                        if hasattr(output, 'request_id') and output.request_id != self.request_id:
                            logger.warning(
                                "Output request_id mismatch: expected %s, got %s",
                                self.request_id, output.request_id
                            )
                            continue

                        yield output

                    except Exception as e:
                        logger.error("Failed to receive output: %s", e)
                        continue

            await self.protocol.egress_loop(output_generator())

        except Exception as e:
            logger.error("Egress task error: %s", e, exc_info=True)
            raise
        finally:
            logger.info("Egress task ended")

    async def _control_task(self) -> None:
        """Route control messages (preserves original control loop logic)"""
        logger.info("Control task started")

        try:
            async for params in self.protocol.control_loop(self._stop_event):
                if self._stop_event.is_set():
                    break

                try:
                    await self.process_manager.update_params(params)
                except Exception as e:
                    logger.error("Failed to update params: %s", e)

        except Exception as e:
            logger.error("Control task error: %s", e, exc_info=True)
            raise
        finally:
            logger.info("Control task ended")

    async def _status_task(self) -> None:
        """Emit status events (preserves original 10s interval)"""
        logger.info("Status task started")

        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=10.0)
                    break  # stop event was set
                except asyncio.TimeoutError:
                    pass  # normal timeout for status emission

                if self.process_manager:
                    try:
                        status = self.process_manager.get_status(clear_transient=True)
                        await self.protocol.emit_monitoring_event(
                            status.model_dump(),
                            queue_event_type="ai_stream_events"
                        )
                    except Exception as e:
                        logger.error("Failed to emit status: %s", e)

        except Exception as e:
            logger.error("Status task error: %s", e, exc_info=True)
            raise
        finally:
            logger.info("Status task ended")

    async def _health_task(self) -> None:
        """Health monitoring with complete original state machine logic"""
        logger.info("Health task started")

        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    pass

                await self._check_health()

        except Exception as e:
            logger.error("Health task error: %s", e, exc_info=True)
            raise
        finally:
            logger.info("Health task ended")

    async def _check_health(self) -> None:
        """Complete health check with original ProcessGuardian state machine logic"""
        if not self.process_manager:
            return

        try:
            current_time = time.time()
            status = self.process_manager.get_status()

            # Compute state using original logic from ProcessGuardian._compute_current_state()
            computed_state = self._compute_current_state(status, current_time)

            if computed_state == status.state:
                return

            # Handle state transitions
            if computed_state == PipelineState.OFFLINE:
                # Revert to initial params when stream stops (original behavior)
                initial_params = self.process_manager.initial_params
                self.process_manager.process.update_params(initial_params)

            elif computed_state == PipelineState.ERROR:
                try:
                    restart_count = status.inference_status.restart_count
                    logger.error(
                        "Pipeline ERROR state, restart_count=%d", restart_count
                    )

                    # Stop stream first
                    self.trigger_stop_stream()

                    # Check restart policy (preserve original limits)
                    if restart_count >= 3:
                        logger.error("Max restarts reached (%d)", restart_count)
                        self.state = SessionState.ERROR
                        self.state_reason = f"Max restarts reached: {restart_count}"
                        self._stop_event.set()
                        return

                    # Skip restart for ComfyUI (preserve original hotfix)
                    if self.pipeline == "comfyui":
                        logger.error("Skipping restart for ComfyUI pipeline")
                        self.state = SessionState.ERROR
                        self.state_reason = "ComfyUI restart disabled"
                        self._stop_event.set()
                        return

                    # Perform restart
                    await self.process_manager.restart_process()

                except Exception as e:
                    logger.exception("Failed to restart process")
                    self.state = SessionState.ERROR
                    self.state_reason = f"Restart failed: {e}"
                    self._stop_event.set()

            # Update state if not ERROR (preserve original logic)
            if computed_state != PipelineState.ERROR:
                status.update_state(computed_state)

        except Exception as e:
            logger.error("Health check failed: %s", e, exc_info=True)

    def _compute_current_state(self, status: PipelineStatus, current_time: float) -> str:
        """Complete original state machine logic from ProcessGuardian._compute_current_state()"""

        if status.state == PipelineState.ERROR:
            return PipelineState.ERROR

        # Process health checks
        if not self.process_manager.process or not self.process_manager.process.is_alive():
            logger.error("Process not alive")
            return PipelineState.ERROR

        if (not self.process_manager.process.is_pipeline_initialized() or
            self.process_manager.process.is_done()):
            return PipelineState.LOADING

        # Stream activity checks
        input_status = status.input_status
        start_time = max(
            self.process_manager.process.start_time if self.process_manager.process else 0,
            status.start_time
        )

        last_input_time = input_status.last_input_time
        if last_input_time:
            last_input_time = last_input_time / 1000.0  # Convert from ms

        time_since_last_input = current_time - (last_input_time or start_time)

        # Stream running checks
        if not self.is_stream_running():
            is_offline = time_since_last_input > 3 or not last_input_time
            return PipelineState.OFFLINE if is_offline else PipelineState.DEGRADED_INPUT

        # Idle shutdown check (preserve original 60/90s timeouts)
        if time_since_last_input > 60:
            if self.trigger_stop_stream():
                logger.info(
                    "Shutting down stream due to inactivity: %.1fs idle",
                    time_since_last_input
                )
            if time_since_last_input < 90:
                return PipelineState.DEGRADED_INPUT
            # Force ERROR after 90s total idle time
            return PipelineState.ERROR

        # Pipeline load state checks
        inference = status.inference_status
        last_output_time = inference.last_output_time
        if last_output_time:
            last_output_time = last_output_time / 1000.0  # Convert from ms

        time_since_last_output = current_time - (last_output_time or 0)
        pipeline_load_time = max(
            inference.last_params_update_time / 1000.0 if inference.last_params_update_time else 0,
            start_time
        )
        time_since_pipeline_load = max(0, current_time - pipeline_load_time - 2)

        active_after_load = time_since_last_output < time_since_pipeline_load

        if not active_after_load:
            is_params_update = (inference.last_params_update_time or 0) > start_time * 1000
            load_grace_period = 2 if is_params_update else 10
            load_timeout = 60 if is_params_update else 120

            if time_since_pipeline_load < load_grace_period:
                return PipelineState.ONLINE
            elif time_since_last_input > time_since_pipeline_load:
                return PipelineState.DEGRADED_INPUT
            elif time_since_pipeline_load < load_timeout:
                return PipelineState.DEGRADED_INFERENCE
            else:
                return PipelineState.ERROR

        # Active stream checks
        stopped_producing_frames = (
            time_since_last_output > (time_since_last_input + 1) and
            time_since_last_output > 5
        )
        if stopped_producing_frames:
            return PipelineState.ERROR

        # Recent error/restart checks
        recent_error = (inference.last_error_time or 0) > (current_time - 15) * 1000
        recent_restart = (inference.last_restart_time or 0) > (current_time - 60) * 1000

        if recent_error or recent_restart:
            return PipelineState.DEGRADED_INFERENCE
        elif time_since_last_input > 2 or input_status.fps < 15:
            return PipelineState.DEGRADED_INPUT
        elif time_since_last_output > 2 or inference.fps < min(10, 0.8 * input_status.fps):
            return PipelineState.DEGRADED_INFERENCE

        return PipelineState.ONLINE
