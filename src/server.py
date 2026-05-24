import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

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
        version="1.0.6",
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

    logger.info("Starting AgentWhetters_general_purple on %s:%d", args.host, args.port)
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
