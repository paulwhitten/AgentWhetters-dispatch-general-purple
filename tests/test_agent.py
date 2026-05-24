import pytest
import httpx


@pytest.fixture
def agent_url(request):
    return request.config.getoption("--agent-url")


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:9009",
        help="URL of the running agent to test against",
    )


@pytest.mark.asyncio
async def test_agent_card(agent_url):
    """Verify the agent card is served at the well-known endpoint."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{agent_url}/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "AgentWhetters_general_purple"
        assert "skills" in card
        assert len(card["skills"]) > 0


@pytest.mark.asyncio
async def test_agent_responds(agent_url):
    """Verify the agent can handle a basic SendMessage request."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            agent_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"text": "Hello, what can you do?"}],
                        "messageId": "test-msg-001",
                    }
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
