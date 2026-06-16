from fastapi.testclient import TestClient


def test_health_endpoint():
    import os

    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
