"""Tests for the Vercel-facing FastAPI application."""

import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app as vercel_app


class FakeResearchGraph:
    """Small async stream that captures request configuration."""

    def __init__(self):
        self.config = None

    async def astream(self, state, config, stream_mode):
        self.config = config
        assert state["messages"][0].content == "对比产品 A 和 B"
        assert stream_mode == "updates"
        yield {"clarify_with_user": {"messages": [AIMessage(content="范围清晰。")]}}
        yield {"final_report_generation": {"final_report": "# 报告\n\n结论"}}


def test_health_and_static_homepage():
    client = TestClient(vercel_app.app)
    assert client.get("/api/health").json() == {
        "status": "ok",
        "service": "competex",
    }
    response = client.get("/")
    assert response.status_code == 200
    assert "CompeteX" in response.text


def test_research_stream_uses_request_scoped_keys(monkeypatch):
    graph = FakeResearchGraph()
    monkeypatch.setattr(vercel_app, "deep_researcher", graph)
    client = TestClient(vercel_app.app)
    response = client.post(
        "/api/research",
        json={
            "messages": [{"role": "user", "content": "对比产品 A 和 B"}],
            "keys": {"deepseek": "request-secret"},
            "search_api": "bing",
            "research_model": "deepseek:deepseek-chat",
            "final_report_model": "deepseek:deepseek-chat",
        },
    )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert events[-1] == {
        "type": "complete",
        "content": "# 报告\n\n结论",
        "status": "complete",
    }
    assert graph.config["configurable"]["apiKeys"] == {
        "DEEPSEEK_API_KEY": "request-secret"
    }
    assert "request-secret" not in response.text
