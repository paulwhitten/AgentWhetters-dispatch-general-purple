import argparse
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from starlette.types import ASGIApp, Receive, Scope, Send

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor

load_dotenv()

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

logging.basicConfig(
    level=os.environ.get("AGENT_LOG_LEVEL", "INFO"),
    format=_LOG_FORMAT,
)

# Add file handler: logs go to ./logs/ AND stdout
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"agent-{datetime.now():%Y%m%d-%H%M%S}.log"
_file_handler = logging.FileHandler(_log_file)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_file_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("agentwhetters")
logger.info("Logging to %s", _log_file)


class MessageIdMiddleware:
    """ASGI middleware that injects messageId/taskId into A2A requests.

    Some benchmark harnesses omit these fields which the a2a-sdk requires.
    Without this patch, the server returns a 400 JSON error instead of SSE.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        body_parts: list[bytes] = []
        request_complete = False

        async def buffered_receive():
            nonlocal request_complete
            if request_complete:
                return {"type": "http.request", "body": b"", "more_body": False}
            msg = await receive()
            body_parts.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                request_complete = True
            return msg

        while not request_complete:
            await buffered_receive()

        body = b"".join(body_parts)
        modified = False

        try:
            data = json.loads(body)
            if (
                isinstance(data, dict)
                and data.get("method") in ("message/send", "message/stream")
                and "params" in data
            ):
                msg = data["params"].get("message", {})
                if isinstance(msg, dict) and "messageId" not in msg:
                    msg["messageId"] = str(uuid.uuid4())
                    modified = True
                config = data["params"].setdefault("configuration", {})
                if isinstance(config, dict) and "taskId" not in config:
                    config["taskId"] = str(uuid.uuid4())
                    modified = True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        if modified:
            body = json.dumps(data).encode()

        body_sent = False

        async def replay_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def main():
    parser = argparse.ArgumentParser(description="AgentWhetters General Purple Agent")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    general_skill = AgentSkill(
        id="general_purpose",
        name="General Purpose Agent",
        description=(
            "A general-purpose agent capable of coding, research, analysis, "
            "web browsing, cybersecurity, game playing, negotiation, and more. "
            "Adapts its strategy based on the task type."
        ),
        tags=[
            "coding", "research", "analysis", "cybersecurity",
            "web", "game", "negotiation", "general",
        ],
        examples=[],
    )

    agent_card = AgentCard(
        name="AgentWhetters_dispatch_general_purple",
        description=(
            "General-purpose purple agent for AgentX-AgentBeats Sprint 4. "
            "Adapts across coding, research, cybersecurity, game, finance, "
            "and other benchmark categories using task-adaptive strategies."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.11",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[general_skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None,
    )

    app = server.build()
    app.add_middleware(MessageIdMiddleware)

    logger.info("Starting AgentWhetters_general_purple on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
