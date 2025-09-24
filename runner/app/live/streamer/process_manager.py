import asyncio
import logging
import time
from typing import Optional

from .process import PipelineProcess
from .status import PipelineState, PipelineStatus, InferenceStatus, InputStatus
from trickle import InputFrame, AudioFrame, VideoFrame, OutputFrame, VideoOutput, AudioOutput
FPS_LOG_INTERVAL = 10.0

class ProcessManager:
    """
    Process manager that handles the AI pipeline subprocess.

    Responsibilities:
    - Spawn and manage the pipeline process
    - Provide I/O interface (send_input, recv_output, update_params)
    - Track measurements and status (FPS, timestamps, errors)
    """

    def __init__(self, pipeline: str, initial_params: dict):
        self.pipeline = pipeline
        self.initial_params = initial_params

        self.process: Optional[PipelineProcess] = None
        self.status = PipelineStatus(pipeline=pipeline, start_time=0).update_params(
            initial_params, False
        )

        self.input_fps_counter = FPSCounter()
        self.output_fps_counter = FPSCounter()

    async def start(self):
        """Start the pipeline process."""
        if self.process is not None:
            raise RuntimeError("ProcessManager already started")

        logging.info(f"Starting ProcessManager for pipeline: {self.pipeline}")
        self.process = PipelineProcess.start(self.pipeline, self.initial_params)
        self.status.update_state(PipelineState.LOADING)

    async def stop(self):
        """Stop the pipeline process."""
        if self.process:
            logging.info("Stopping ProcessManager")
            await self.process.stop()
            self.process = None

    async def reset_stream(
        self,
        request_id: str,
        manifest_id: str,
        stream_id: str,
        params: dict
    ):
        """Reset for a new stream session."""
        if not self.process:
            raise RuntimeError("Process not running")

        logging.info(f"Resetting stream: request_id={request_id}")

        # Reset status and counters for new stream
        self.status.start_time = time.time()
        self.status.input_status = InputStatus()
        self.input_fps_counter.reset()
        self.output_fps_counter.reset()

        # Reset the subprocess for the new session
        self.process.reset_stream(request_id, manifest_id, stream_id)
        await self.update_params(params)
        self.status.update_state(PipelineState.ONLINE)

    def send_input(self, frame: InputFrame):
        """Send input frame to the process."""
        if not self.process:
            raise RuntimeError("Process not running")

        # Update input tracking
        self.status.input_status.last_input_time = time.time()
        self.input_fps_counter.inc()

        self.process.send_input(frame)

    async def recv_output(self) -> OutputFrame | None:
        """Receive output frame from the process."""
        if not self.process:
            raise RuntimeError("Process not running")

        output = await self.process.recv_output()

        # Update output tracking (only for non-loading frames)
        if output and not output.is_loading_frame:
            self.status.inference_status.last_output_time = time.time()
            self.output_fps_counter.inc()

        return output

    async def update_params(self, params: dict):
        """Update pipeline parameters."""
        if not self.process:
            raise RuntimeError("Process not running")

        # Forward to process
        self.process.update_params(params)

        # Update status tracking
        self.status.update_params(params)

        logging.info(
            f"ProcessManager: Parameter update queued. "
            f"hash={self.status.inference_status.last_params_hash} params={params}"
        )

    def get_status(self, clear_transient: bool = False) -> PipelineStatus:
        """Get current pipeline status."""
        if not self.process:
            # Return basic status if process not running
            return PipelineStatus(
                pipeline=self.pipeline,
                start_time=0,
                state=PipelineState.OFFLINE
            )

        # Update FPS measurements
        current_time = time.time()
        input_fps = self.input_fps_counter.fps(current_time)
        output_fps = self.output_fps_counter.fps(current_time)

        if input_fps is not None:
            self.status.input_status.fps = input_fps
        if output_fps is not None:
            self.status.inference_status.fps = output_fps

        # Log FPS periodically
        if input_fps is not None and input_fps > 0:
            logging.info(f"Input FPS: {input_fps:.2f}")
        if output_fps is not None and output_fps > 0:
            logging.info(f"Output FPS: {output_fps:.2f}")

        # Compute current state based on measurements
        computed_state = self._compute_current_state()
        if computed_state != self.status.state:
            self.status.update_state(computed_state)

        # Return copy with optional transient field clearing
        status = self.status.model_copy(deep=True)
        if clear_transient:
            status.inference_status.last_params = None
            status.inference_status.last_restart_logs = None

        return status

    def _compute_current_state(self) -> str:
        """
        Compute current pipeline state based on measurements only.

        Policy decisions (what to do about states) are handled by StreamSession.
        This method only reports what we observe.
        """
        if not self.process:
            return PipelineState.OFFLINE

        # Process health checks
        if not self.process.is_alive():
            return PipelineState.ERROR

        if not self.process.is_pipeline_initialized() or self.process.done.is_set():
            return PipelineState.LOADING

        current_time = time.time()
        input_status = self.status.input_status
        inference_status = self.status.inference_status
        start_time = max(self.process.start_time, self.status.start_time)

        # Time calculations
        time_since_start = current_time - start_time
        time_since_last_input = current_time - (input_status.last_input_time or start_time)
        time_since_last_output = current_time - (inference_status.last_output_time or 0)
        time_since_params_update = current_time - (inference_status.last_params_update_time or start_time)

        # Check for recent activity after pipeline load/params update
        pipeline_load_time = max(inference_status.last_params_update_time or 0, start_time)
        time_since_pipeline_load = max(0, current_time - pipeline_load_time - 2)  # -2s buffer
        active_after_load = time_since_last_output < time_since_pipeline_load

        # Loading/initialization states
        if not active_after_load:
            is_params_update = (inference_status.last_params_update_time or 0) > start_time
            load_grace_period = 2 if is_params_update else 10
            load_timeout = 60 if is_params_update else 120

            if time_since_pipeline_load < load_grace_period:
                return PipelineState.ONLINE
            elif time_since_last_input > time_since_pipeline_load:
                return PipelineState.DEGRADED_INPUT
            elif time_since_pipeline_load < load_timeout:
                return PipelineState.DEGRADED_INFERENCE
            else:
                return PipelineState.ERROR  # Not starting after timeout

        # Active stream health checks
        stopped_producing_frames = (
            time_since_last_output > (time_since_last_input + 1)
            and time_since_last_output > 5
        )
        if stopped_producing_frames:
            return PipelineState.ERROR

        # Degraded states based on recent errors/restarts
        recent_error = (inference_status.last_error_time or 0) > current_time - 15
        recent_restart = (inference_status.last_restart_time or 0) > current_time - 60
        if recent_error or recent_restart:
            return PipelineState.DEGRADED_INFERENCE

        # Degraded states based on input/output performance
        if time_since_last_input > 2 or input_status.fps < 15:
            return PipelineState.DEGRADED_INPUT
        elif time_since_last_output > 2 or inference_status.fps < min(10, 0.8 * input_status.fps):
            return PipelineState.DEGRADED_INFERENCE

        return PipelineState.ONLINE

    def get_last_error(self) -> tuple[str, float] | None:
        """Get the most recent error from the process, if any."""
        if not self.process:
            return None

        last_error = self.process.get_last_error()
        if last_error:
            error_msg, error_time = last_error
            # Update our status tracking
            self.status.inference_status.last_error = error_msg
            self.status.inference_status.last_error_time = error_time
            return last_error

        return None

    def is_alive(self) -> bool:
        """Check if the process is alive."""
        return self.process is not None and self.process.is_alive()

    def is_pipeline_initialized(self) -> bool:
        """Check if the pipeline is initialized."""
        return self.process is not None and self.process.is_pipeline_initialized()


class FPSCounter:
    """Helper class for tracking FPS measurements."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the counter to initial state."""
        self.frame_count = 0
        self.start_time = None
        self.last_fps_time = 0

    def inc(self):
        """Increment frame count."""
        current_time = time.time()
        if not self.start_time:
            # First frame sets the measurement window
            self.start_time = current_time
            self.frame_count = 0
            self.last_fps_time = current_time
        else:
            self.frame_count += 1

    def fps(self, now: float = None) -> Optional[float]:
        """
        Calculate current FPS and reset measurement window.
        Returns None if no frames have been processed yet.
        """
        if now is None:
            now = time.time()

        if not self.frame_count or not self.start_time:
            return None

        # Only report FPS if enough time has passed for meaningful measurement
        if now - self.last_fps_time < FPS_LOG_INTERVAL:
            return None

        fps = self.frame_count / (now - self.start_time)

        # Reset measurement window
        self.start_time = now
        self.frame_count = 0
        self.last_fps_time = now

        return fps
