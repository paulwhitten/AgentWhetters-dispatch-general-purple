import pytest
import httpx


@pytest.fixture
def agent_url(request):
    return request.config.getoption("--agent-url")


@pytest.mark.asyncio
async def test_agent_card(agent_url):
    """Verify the agent card is served at the well-known endpoint."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{agent_url}/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "AgentWhetters_dispatch_general_purple"
        assert "skills" in card
        assert len(card["skills"]) > 0


@pytest.mark.asyncio
async def test_agent_responds(agent_url):
    """Verify the agent can handle a basic message/send request."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            agent_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Hello, what can you do?"}],
                        "messageId": "test-msg-001",
                    }
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
