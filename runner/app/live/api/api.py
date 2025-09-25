import asyncio
import logging
import json
import os
import tempfile
import time
from typing import Optional, cast, Dict, Any

from aiohttp import web
from pydantic import BaseModel, Field
from typing import Annotated

from session import StreamSession
from process_manager import ProcessManager
from streamer.protocol.trickle import TrickleProtocol
from streamer.process import config_logging
from trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT

MAX_FILE_AGE = 86400  # 1 day

# File to store the last params that a stream was started with. Used to cleanup
# left over resources (e.g. trickle channels) left by a crashed process.
last_params_file = os.path.join(tempfile.gettempdir(), "ai_runner_last_params.json")

class StartStreamParams(BaseModel):
    subscribe_url: Annotated[
        str,
        Field(
            ...,
            description="Source URL of the incoming stream to subscribe to.",
        ),
    ]
    publish_url: Annotated[
        str,
        Field(
            ...,
            description="Destination URL of the outgoing stream to publish.",
        ),
    ]
    control_url: Annotated[
        str,
        Field(
            default="",
            description="URL for subscribing via Trickle protocol for updates in the live video-to-video generation params.",
        ),
    ]
    events_url: Annotated[
        str,
        Field(
            default="",
            description="URL for publishing events via Trickle protocol for pipeline status and logs.",
        ),
    ]
    params: Annotated[
        Dict,
        Field(default={}, description="Initial parameters for the pipeline."),
    ]
    request_id: Annotated[
        str,
        Field(default="", description="Unique identifier for the request."),
    ]
    manifest_id: Annotated[
        str,
        Field(default="", description="Orchestrator identifier for the request."),
    ]
    stream_id: Annotated[
        str,
        Field(default="", description="Unique identifier for the stream."),
    ]

async def cleanup_last_stream():
    """Clean up resources from previous crashed stream"""
    if not os.path.exists(last_params_file):
        logging.debug("No last stream params found to cleanup")
        return

    try:
        with open(last_params_file, "r") as f:
            params = StartStreamParams(**json.load(f))
        os.remove(last_params_file)

        logging.info(
            "Cleaning up last stream trickle channels for request_id=%s subscribe_url=%s publish_url=%s control_url=%s events_url=%s",
            params.request_id, params.subscribe_url, params.publish_url,
            params.control_url, params.events_url
        )
        protocol = TrickleProtocol(
            params.subscribe_url,
            params.publish_url,
            params.control_url,
            params.events_url,
        )
        # Start and stop the protocol immediately to close trickle channels
        await protocol.start()
        await protocol.stop()

    except Exception:
        logging.exception("Error cleaning up last stream trickle channels")

async def parse_request_data(request: web.Request) -> Dict[str, Any]:
    """Parse JSON request data with error handling"""
    if request.content_type.startswith("application/json"):
        return await request.json()
    else:
        raise ValueError(f"Unknown content type: {request.content_type}")

async def handle_start_stream(request: web.Request):
    """Handle POST /api/live-video-to-video - Start streaming session"""
    try:
        stream_request_timestamp = int(time.time() * 1000)
        process_manager = cast(ProcessManager, request.app["process_manager"])
        prev_session = cast(Optional[StreamSession], request.app.get("session"))

        # Stop previous session if running
        if prev_session and prev_session.is_running():
            try:
                logging.info("Stopping previous streaming session")
                await prev_session.stop(timeout=10)
            except asyncio.TimeoutError as e:
                logging.error("Timeout stopping previous session: %s", e)
                raise web.HTTPBadRequest(text="Timeout stopping previous session")
            except Exception as e:
                logging.error("Error stopping previous session: %s", e)
                raise web.HTTPInternalServerError(text=f"Error stopping previous session: {e}")

        # Parse and validate request parameters
        try:
            params_data = await parse_request_data(request)
            params = StartStreamParams(**params_data)
        except Exception as e:
            logging.error("Invalid request parameters: %s", e)
            raise web.HTTPBadRequest(text=f"Invalid parameters: {e}")

        # Save params for crash recovery
        try:
            with open(last_params_file, "w") as f:
                json.dump(params.model_dump(), f)
        except Exception as e:
            logging.error("Error saving last params to file: %s", e)

        # Configure logging context
        config_logging(
            request_id=params.request_id,
            manifest_id=params.manifest_id,
            stream_id=params.stream_id
        )

        # Handle dimension configuration
        width = params.params.get("width", DEFAULT_WIDTH)
        height = params.params.get("height", DEFAULT_HEIGHT)

        # TODO: Remove this once ComfyUI pipeline supports different resolutions without restart
        if process_manager.pipeline == "comfyui":
            width = height = 512
            params.params = params.params | {"width": width, "height": height}
            logging.warning("Using default 512x512 dimensions for ComfyUI pipeline")
        else:
            logging.info("Using dimensions from params: %dx%d", width, height)

        # Create protocol adapter
        protocol = TrickleProtocol(
            params.subscribe_url,
            params.publish_url,
            params.control_url,
            params.events_url,
            width,
            height,
        )

        # Create new StreamSession with ProcessManager already created
        session = StreamSession(
            pipeline=process_manager.pipeline,
            protocol=protocol,
            request_id=params.request_id,
            manifest_id=params.manifest_id,
            stream_id=params.stream_id,
        )

        # Inject the existing ProcessManager into the session
        session.process_manager = process_manager

        # Start the streaming session
        try:
            await session.start(params.params)
            request.app["session"] = session

            # Emit stream trace event
            await protocol.emit_monitoring_event({
                "type": "runner_receive_stream_request",
                "timestamp": stream_request_timestamp,
            }, queue_event_type="stream_trace")

        except Exception as e:
            logging.error("Failed to start streaming session: %s", e)
            # Clean up session on failure
            try:
                await session.stop()
            except Exception:
                pass
            raise web.HTTPInternalServerError(text=f"Failed to start streaming session: {e}")

        return web.Response(text="Stream started successfully")

    except web.HTTPError:
        # Re-raise HTTP errors as-is
        raise
    except Exception as e:
        logging.error("Unexpected error starting stream: %s", e, exc_info=True)
        return web.HTTPInternalServerError(text=f"Internal server error: {e}")


async def handle_params_update(request: web.Request):
    """Handle POST /api/params - Update pipeline parameters"""
    try:
        # Parse parameters
        try:
            params = await parse_request_data(request)
        except Exception as e:
            logging.error("Invalid parameter update request: %s", e)
            raise web.HTTPBadRequest(text=f"Invalid parameters: {e}")

        # Get ProcessManager (always available)
        process_manager = cast(ProcessManager, request.app["process_manager"])

        # Update parameters via ProcessManager
        try:
            await process_manager.update_params(params)
        except Exception as e:
            logging.error("Failed to update parameters: %s", e)
            raise web.HTTPInternalServerError(text=f"Failed to update parameters: {e}")

        return web.Response(text="Params updated successfully")

    except web.HTTPError:
        # Re-raise HTTP errors as-is
        raise
    except Exception as e:
        logging.error("Unexpected error updating params: %s", e, exc_info=True)
        return web.HTTPInternalServerError(text=f"Internal server error: {e}")


async def handle_get_status(request: web.Request):
    """Handle GET /api/status - Get pipeline status"""
    try:
        process_manager = cast(ProcessManager, request.app["process_manager"])
        session = cast(Optional[StreamSession], request.app.get("session"))

        # Get base status from ProcessManager
        status = process_manager.get_status()
        status_dict = status.model_dump()

        # Augment with session information if available
        if session:
            status_dict["session_state"] = session.state.value
            if session.state_reason:
                status_dict["session_reason"] = session.state_reason
            status_dict["session_running"] = session.is_running()
        else:
            status_dict["session_state"] = "NO_SESSION"
            status_dict["session_running"] = False

        return web.json_response(status_dict)

    except Exception as e:
        logging.error("Error getting status: %s", e, exc_info=True)
        return web.HTTPInternalServerError(text=f"Failed to get status: {e}")


async def start_http_server(
    port: int,
    process_manager: ProcessManager,
    session: Optional[StreamSession] = None
):
    """Start HTTP server with ProcessManager and optional StreamSession"""

    # Clean up previous crashed streams on startup
    asyncio.create_task(cleanup_last_stream())

    app = web.Application()
    app["process_manager"] = process_manager
    app["session"] = session  # May be None if not streaming

    # Register routes
    app.router.add_post("/api/live-video-to-video", handle_start_stream)
    app.router.add_post("/api/params", handle_params_update)
    app.router.add_get("/api/status", handle_get_status)

    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logging.info("HTTP server started on port %d", port)
    return runner
