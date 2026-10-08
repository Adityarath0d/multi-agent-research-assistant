from fastapi.testclient import TestClient

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.main import create_app


def test_health_returns_ok() -> None:
    app = create_app(Settings(environment="test"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "research-assistant",
        "environment": "test",
    }
