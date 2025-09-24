import asyncio
import argparse
import json
import logging

from lifecycle_controller import LifecycleController
from pipeline_streamer import PipelineStreamer
from process_guardian import ProcessGuardian
from protocols.trickle import TrickleProtocol
from protocols.zeromq import ZeroMQProtocol
from api.api import start_http_server


logger = logging.getLogger("infer")


def parse_args():
    parser = argparse.ArgumentParser("ai-runner infer")
    parser.add_argument("--pipeline", required=True, help="Pipeline name or path")
    parser.add_argument("--params", default="{}", help="JSON string of pipeline params")
    parser.add_argument("--stream-protocol", choices=["trickle", "zeromq"], default="trickle")
    parser.add_argument("--subscribe-url", required=True, help="Inbound frames URL")
    parser.add_argument("--publish-url", required=True, help="Outbound frames URL")
    parser.add_argument("--control-url", help="Control channel URL")
    parser.add_argument("--events-url", help="Events channel URL")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--request-id", default="req-unknown")
    parser.add_argument("--stream-id", default="stream-unknown")
    return parser.parse_args()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()

    try:
        params = json.loads(args.params) if isinstance(args.params, str) else (args.params or {})
    except Exception as e:
        logger.error("Failed parsing params JSON: %s", e)
        params = {}

    guardian = ProcessGuardian(pipeline=args.pipeline, params=params)

    if args.stream_protocol == "trickle":
        protocol = TrickleProtocol(
            subscribe_url=args.subscribe_url,
            publish_url=args.publish_url,
            control_url=args.control_url,
            events_url=args.events_url,
            request_id=args.request_id,
            stream_id=args.stream_id,
        )
    else:
        protocol = ZeroMQProtocol(
            subscribe_url=args.subscribe_url,
            publish_url=args.publish_url,
            control_url=args.control_url,
            events_url=args.events_url,
            request_id=args.request_id,
            stream_id=args.stream_id,
        )

    streamer = PipelineStreamer(protocol=protocol, process=guardian)

    controller = LifecycleController(components=[protocol, guardian, streamer])

    api = await start_http_server(args.http_port, guardian, streamer)

    await controller.run_until_signal()

    await api.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
