import asyncio
import logging
import json
import os
import tempfile
import time
from typing import Optional, cast

from aiohttp import web
from pydantic import BaseModel, Field
from typing import Annotated, Dict

from streamer import ProcessManager, StreamSession
from streamer.protocol.trickle import TrickleProtocol
from streamer.protocol.zeromq import ZeroMQProtocol
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
    """Clean up any leftover trickle channels from previous crashed sessions."""
    if not os.path.exists(last_params_file):
        logging.debug("No last stream params found to cleanup")
        return

    try:
        with open(last_params_file, "r") as f:
            params = StartStreamParams(**json.load(f))
        os.remove(last_params_file)

        logging.info(
            f"Cleaning up last stream trickle channels for request_id={params.request_id} subscribe_url={params.subscribe_url} publish_url={params.publish_url} control_url={params.control_url} events_url={params.events_url}")
        protocol = TrickleProtocol(
            params.subscribe_url,
            params.publish_url,
            params.control_url,
            params.events_url,
        )
        # Start and stop the protocol immediately to make sure trickle channels are closed.
        await protocol.start()
        await protocol.stop()
    except:
        logging.exception(f"Error cleaning up last stream trickle channels")


async def parse_request_data(request: web.Request) -> Dict:
    if request.content_type.startswith("application/json"):
        return await request.json()
    else:
        raise ValueError(f"Unknown content type: {request.content_type}")


async def handle_start_stream(request: web.Request):
    """Handle POST /api/live-video-to-video - start a new streaming session."""
    try:
        stream_request_timestamp = int(time.time() * 1000)
        process_manager = cast(ProcessManager, request.app["process_manager"])
        prev_session = cast(Optional[StreamSession], request.app.get("session"))

        # Stop previous session if running
        if prev_session and prev_session.is_running():
            try:
                logging.info("Stopping previous session")
                await prev_session.stop(timeout=10)
            except asyncio.TimeoutError as e:
                logging.error(f"Timeout stopping previous session: {e}")
                raise web.HTTPBadRequest(text="Timeout stopping previous session")

        # Parse request parameters
        params_data = await parse_request_data(request)
        params = StartStreamParams(**params_data)

        # Save params for crash recovery
        try:
            with open(last_params_file, "w") as f:
                json.dump(params.model_dump(), f)
        except Exception as e:
            logging.error(f"Error saving last params to file: {e}")

        # Configure logging for this stream
        config_logging(request_id=params.request_id, manifest_id=params.manifest_id, stream_id=params.stream_id)

        # Get dimensions from workflow
        width = params.params.get("width", DEFAULT_WIDTH)
        height = params.params.get("height", DEFAULT_HEIGHT)
        logging.info(f"Using dimensions from params: {width}x{height}")
        # Create protocol adapter
        protocol = TrickleProtocol(
            params.subscribe_url,
            params.publish_url,
            params.control_url,
            params.events_url,
            width,
            height,
        )

        # Create new streaming session
        session = StreamSession(
            protocol,
            process_manager,
            params.request_id,
            params.manifest_id,
            params.stream_id,
        )

        # Start the session
        await session.start(params.params)
        request.app["session"] = session

        # Emit stream request trace event (preserved from original)
        # Note: The session already emits this in session.start(), but we emit again
        # here to maintain exact compatibility with the original API behavior
        await session._emit_monitoring_event({
            "type": "runner_receive_stream_request",
            "timestamp": stream_request_timestamp,
        }, queue_event_type="stream_trace")

        return web.Response(text="Stream started successfully")

    except Exception as e:
        logging.error(f"Error starting stream: {e}")
        return web.Response(text=f"Error starting stream: {str(e)}", status=400)


async def handle_params_update(request: web.Request):
    """Handle POST /api/params - update parameters for the current session."""
    try:
        params = await parse_request_data(request)

        session = cast(Optional[StreamSession], request.app.get("session"))
        if not session or not session.is_running():
            # If no active session, update the process manager directly
            # This maintains compatibility with direct process management
            process_manager = cast(ProcessManager, request.app["process_manager"])
            await process_manager.update_params(params)
        else:
            # Update through the active session
            await session.update_params(params)

        return web.Response(text="Params updated successfully")

    except Exception as e:
        logging.error(f"Error updating params: {e}")
        return web.Response(text=f"Error updating params: {str(e)}", status=400)


async def handle_get_status(request: web.Request):
    """Handle GET /api/status - return current pipeline status."""
    try:
        process_manager = cast(ProcessManager, request.app["process_manager"])
        session = cast(Optional[StreamSession], request.app.get("session"))

        # Get status from process manager
        status = process_manager.get_status()
        status_dict = status.model_dump()

        # Optionally augment with session state if needed
        # For now, we preserve the exact same status format to avoid breaking clients
        if session:
            # We could add session state info here, but keeping it simple for now
            # status_dict["session_state"] = session.get_state().value
            pass

        return web.json_response(status_dict)

    except Exception as e:
        logging.error(f"Error getting status: {e}")
        return web.Response(text=f"Error getting status: {str(e)}", status=500)


async def start_http_server(
    port: int,
    process_manager: ProcessManager,
    initial_session: Optional[StreamSession] = None
):
    """Start the HTTP server with the process manager and optional initial session."""
    # Clean up any leftover resources from previous runs
    asyncio.create_task(cleanup_last_stream())

    app = web.Application()
    app["process_manager"] = process_manager
    app["session"] = initial_session  # May be None

    # Route handlers - same paths as before
    app.router.add_post("/api/live-video-to-video", handle_start_stream)
    app.router.add_post("/api/params", handle_params_update)
    app.router.add_get("/api/status", handle_get_status)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"HTTP server started on port {port}")
    return runner
