import asyncio
import logging
import time
from typing import Optional, Dict, Any

from streamer.process import PipelineProcess
from streamer.status import PipelineState, PipelineStatus, InferenceStatus, InputStatus
from trickle import InputFrame, OutputFrame

FPS_LOG_INTERVAL = 10.0

logger = logging.getLogger(__name__)


class FPSCounter:
    """Frame rate counter for monitoring input/output throughput"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset counter to initial state - call between streams"""
        self.frame_count = 0
        self.start_time = None

    def inc(self):
        """Increment frame count"""
        if not self.start_time:
            # First frame sets measurement window start time
            self.start_time = time.time()
            self.frame_count = 0
        else:
            self.frame_count += 1

    def fps(self, now: float = None) -> float:
        """Calculate current FPS and reset measurement window"""
        if now is None:
            now = time.time()

        if not self.frame_count or not self.start_time:
            fps = 0.0
        else:
            fps = self.frame_count / (now - self.start_time)

        # Start next measuring window immediately
        self.start_time = now
        self.frame_count = 0
        return fps


class ProcessManager:
    """
    Complete subprocess management with all original ProcessGuardian functionality.
    Handles pipeline subprocess, I/O, status tracking, and process restart logic.
    """

    def __init__(self, pipeline: str, initial_params: Dict[str, Any]):
        self.pipeline = pipeline
        self.initial_params = initial_params or {}

        # Subprocess management
        self.process: Optional[PipelineProcess] = None

        # Status tracking
        self.status = PipelineStatus(
            pipeline=pipeline,
            start_time=0
        ).update_params(self.initial_params, False)

        # Metrics
        self.input_fps_counter = FPSCounter()
        self.output_fps_counter = FPSCounter()

        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
        self._stop_monitoring = asyncio.Event()

        # Callbacks interface (for StreamSession compatibility)
        self.streamer_callbacks = None

    async def start(self) -> None:
        """Start the subprocess and monitoring"""
        if self.process and self.process.is_alive():
            logger.warning("Process already running")
            return

        try:
            logger.info("Starting ProcessManager: pipeline=%s", self.pipeline)

            self.process = PipelineProcess.start(self.pipeline, self.initial_params)
            self.status.update_state(PipelineState.LOADING)

            # Start monitoring task for FPS and error collection
            self._stop_monitoring.clear()
            self.monitor_task = asyncio.create_task(
                self._monitor_loop(), name="process_monitor"
            )

            logger.info("ProcessManager started: pipeline=%s", self.pipeline)

        except Exception as e:
            logger.error("Failed to start ProcessManager: %s", e, exc_info=True)
            await self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop subprocess and monitoring"""
        logger.info("Stopping ProcessManager: pipeline=%s", self.pipeline)

        # Stop monitoring
        self._stop_monitoring.set()
        if self.monitor_task:
            try:
                await asyncio.wait_for(self.monitor_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Monitor task stop timed out, cancelling")
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            self.monitor_task = None

        # Stop subprocess
        await self._cleanup()

        logger.info("ProcessManager stopped: pipeline=%s", self.pipeline)

    async def reset_stream(
        self,
        request_id: str,
        manifest_id: str,
        stream_id: str,
        params: Dict[str, Any],
        streamer_callbacks=None
    ) -> None:
        """Reset stream state for new session with callbacks"""
        if not self.process:
            raise RuntimeError("Process not running")

        logger.info(
            "Resetting stream: request_id=%s manifest_id=%s stream_id=%s",
            request_id, manifest_id, stream_id
        )

        # Store callbacks for compatibility
        self.streamer_callbacks = streamer_callbacks

        # Reset status and counters
        self.status.start_time = time.time()
        self.status.input_status = InputStatus()
        self.input_fps_counter.reset()
        self.output_fps_counter.reset()

        # Reset subprocess state
        self.process.reset_stream(request_id, manifest_id, stream_id)

        # Update parameters
        await self.update_params(params)

        # Mark as online
        self.status.update_state(PipelineState.ONLINE)

    def send_input(self, frame: InputFrame) -> None:
        """Send input frame to subprocess"""
        if not self.process:
            raise RuntimeError("Process not running")

        # Update metrics
        self.status.input_status.last_input_time = time.time()
        self.input_fps_counter.inc()

        # Forward to subprocess
        self.process.send_input(frame)

    async def recv_output(self) -> Optional[OutputFrame]:
        """Receive output frame from subprocess"""
        if not self.process:
            raise RuntimeError("Process not running")

        output = await self.process.recv_output()

        if output and not output.is_loading_frame:
            # Update metrics for non-loading frames
            self.status.inference_status.last_output_time = time.time()
            self.output_fps_counter.inc()

        return output

    async def update_params(self, params: Dict[str, Any]) -> None:
        """Update pipeline parameters with event emission"""
        if not self.process:
            raise RuntimeError("Process not running")

        logger.info(
            "ProcessManager updating params: pipeline=%s params=%s",
            self.pipeline, params
        )

        # Forward to subprocess
        self.process.update_params(params)

        # Update status
        self.status.update_params(params)

        # Emit event if callbacks available (preserves original behavior but commented out)
        # The original code had this commented out to avoid event spam
        # if self.streamer_callbacks:
        #     await self.streamer_callbacks.emit_monitoring_event({
        #         "type": "params_update",
        #         "pipeline": self.pipeline,
        #         "params": params,
        #         "params_hash": self.status.inference_status.last_params_hash,
        #         "update_time": self.status.inference_status.last_params_update_time,
        #     })

        # Log the update (preserve original logging)
        logger.info(
            "ProcessManager parameter update queued: hash=%s",
            self.status.inference_status.last_params_hash
        )

    def get_status(self, clear_transient: bool = False) -> PipelineStatus:
        """Get current pipeline status snapshot"""
        status = self.status.model_copy(deep=True)

        if clear_transient:
            # Clear large transient fields but return them in the copy
            self.status.inference_status.last_params = None
            self.status.inference_status.last_restart_logs = None

        return status

    async def restart_process(self) -> None:
        """Restart subprocess with complete original logic from ProcessGuardian"""
        if not self.process:
            raise RuntimeError("Process not started")

        logger.info("Restarting ProcessManager subprocess")

        try:
            # Capture logs before stopping (preserve original behavior)
            restart_logs = self.process.get_recent_logs()

            # Stop current process
            await self.process.stop()

            # Start new process
            self.process = PipelineProcess.start(self.pipeline, self.initial_params)
            self.status.update_state(PipelineState.LOADING)

            # Update restart status
            curr_status = self.status.inference_status
            self.status.inference_status = InferenceStatus(
                restart_count=curr_status.restart_count + 1,
                last_restart_time=time.time(),
                last_restart_logs=restart_logs,
            )

            # Emit restart event if callbacks available
            if self.streamer_callbacks:
                await self.streamer_callbacks.emit_monitoring_event({
                    "type": "restart",
                    "pipeline": self.pipeline,
                    "restart_count": self.status.inference_status.restart_count,
                    "restart_time": self.status.inference_status.last_restart_time,
                    "restart_logs": restart_logs,
                })

            logger.info(
                "ProcessManager restarted: restart_count=%d",
                self.status.inference_status.restart_count
            )

        except Exception as e:
            logger.error("Failed to restart process: %s", e, exc_info=True)
            raise

    async def _cleanup(self) -> None:
        """Clean up subprocess resources"""
        if self.process:
            try:
                await self.process.stop()
            except Exception as e:
                logger.error("Error stopping process: %s", e)
            finally:
                self.process = None

    async def _monitor_loop(self) -> None:
        """
        Complete monitoring loop with all original functionality.
        Handles FPS calculation, error collection, and event emission.
        """
        logger.info("Process monitor loop started")
        last_fps_compute = time.time()

        try:
            while not self._stop_monitoring.is_set():
                try:
                    # Wait for next monitoring interval
                    await asyncio.wait_for(
                        self._stop_monitoring.wait(),
                        timeout=1.0
                    )
                    break  # Stop event was set

                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue monitoring

                if not self.process:
                    continue

                # Collect errors from subprocess (preserve original logic)
                last_error = self.process.get_last_error()
                if last_error:
                    error_msg, error_time = last_error
                    self.status.inference_status.last_error = error_msg
                    self.status.inference_status.last_error_time = error_time

                    # Emit error event if callbacks available
                    if self.streamer_callbacks:
                        await self.streamer_callbacks.emit_monitoring_event({
                            "type": "error",
                            "pipeline": self.pipeline,
                            "message": error_msg,
                            "time": error_time,
                        })

                    logger.warning(
                        "Process error reported: %s (time=%s)",
                        error_msg, error_time
                    )

                # Update FPS counters periodically (preserve original 10s interval)
                now = time.time()
                if now - last_fps_compute > FPS_LOG_INTERVAL:
                    input_fps = self.input_fps_counter.fps(now)
                    output_fps = self.output_fps_counter.fps(now)

                    self.status.input_status.fps = input_fps
                    self.status.inference_status.fps = output_fps
                    last_fps_compute = now

                    # Log FPS if stream is active and has activity
                    if (self.streamer_callbacks and
                        self.streamer_callbacks.is_stream_running()):
                        if input_fps > 0:
                            logger.info("Input FPS: %.2f", input_fps)
                        if output_fps > 0:
                            logger.info("Output FPS: %.2f", output_fps)

        except asyncio.CancelledError:
            logger.info("Process monitor loop cancelled")
            raise
        except Exception as e:
            logger.error("Process monitor loop error: %s", e, exc_info=True)
        finally:
            logger.info("Process monitor loop ended")
