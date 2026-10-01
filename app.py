"""Vercel-compatible API and static frontend for CompeteX."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, SecretStr

from open_deep_research.deep_researcher import deep_researcher

ROOT = Path(__file__).resolve().parent


class ClientMessage(BaseModel):
    """A conversation message supplied by the browser."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=50_000)


class ApiKeys(BaseModel):
    """Bring-your-own keys used only for the current request."""

    deepseek: SecretStr | None = None
    openai: SecretStr | None = None
    anthropic: SecretStr | None = None
    google: SecretStr | None = None
    tavily: SecretStr | None = None


class ResearchRequest(BaseModel):
    """Configuration accepted by the public research endpoint."""

    messages: list[ClientMessage] = Field(min_length=1, max_length=30)
    keys: ApiKeys
    search_api: Literal["bing", "tavily", "none"] = "bing"
    research_model: str = Field(default="deepseek:deepseek-chat", max_length=120)
    final_report_model: str = Field(default="deepseek:deepseek-chat", max_length=120)
    allow_clarification: bool = True


app = FastAPI(
    title="CompeteX Research API",
    description="Bring-your-own-key competitor research powered by LangGraph.",
    version="1.0.0",
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Return a lightweight readiness response."""
    return {"status": "ok", "service": "competex"}


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None


def _message_text(value: object) -> str:
    if isinstance(value, BaseMessage):
        return _message_text(value.content)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_message_text(item) for item in value)
    if isinstance(value, dict):
        return _message_text(value.get("text") or value.get("content") or "")
    return ""


def _extract_text(output: object) -> tuple[str, bool]:
    """Return visible text and whether it is a final report."""
    if not isinstance(output, dict):
        return "", False
    report = output.get("final_report")
    if isinstance(report, str) and report.strip():
        return report, True
    messages = output.get("messages")
    if isinstance(messages, list) and messages:
        return _message_text(messages[-1]), False
    return "", False


def _line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


@app.post("/api/research")
async def research(request: ResearchRequest) -> StreamingResponse:
    """Run the LangGraph workflow and stream newline-delimited progress events."""

    async def event_stream() -> AsyncIterator[bytes]:
        api_keys = {
            "DEEPSEEK_API_KEY": _secret(request.keys.deepseek),
            "OPENAI_API_KEY": _secret(request.keys.openai),
            "ANTHROPIC_API_KEY": _secret(request.keys.anthropic),
            "GOOGLE_API_KEY": _secret(request.keys.google),
            "TAVILY_API_KEY": _secret(request.keys.tavily),
        }
        api_keys = {key: value for key, value in api_keys.items() if value}
        state_messages = [
            HumanMessage(content=item.content)
            if item.role == "user"
            else AIMessage(content=item.content)
            for item in request.messages
        ]
        config = {
            "configurable": {
                "apiKeys": api_keys,
                "search_api": request.search_api,
                "research_model": request.research_model,
                "compression_model": request.research_model,
                "summarization_model": request.research_model,
                "final_report_model": request.final_report_model,
                "allow_clarification": request.allow_clarification,
                "max_concurrent_research_units": 5,
                "max_researcher_iterations": 2,
                "max_react_tool_calls": 4,
            }
        }
        last_text = ""
        final_report = False

        try:
            yield _line({"type": "started"})
            async for update in deep_researcher.astream(
                {"messages": state_messages},
                config=config,
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue
                for node, output in update.items():
                    yield _line({"type": "progress", "node": node, "status": "done"})
                    text, is_report = _extract_text(output)
                    if text:
                        last_text = text
                        final_report = is_report
                        yield _line(
                            {
                                "type": "result" if is_report else "message",
                                "content": text,
                            }
                        )

            if not last_text:
                raise RuntimeError("研究流程没有返回可展示的结果")
            yield _line(
                {
                    "type": "complete",
                    "content": last_text,
                    "status": "complete" if final_report else "awaiting_input",
                }
            )
        except Exception as exc:  # The stream must report errors after headers are sent.
            error_message = f"{type(exc).__name__}: {exc}"
            for secret in api_keys.values():
                error_message = error_message.replace(secret, "[redacted]")
            yield _line(
                {
                    "type": "error",
                    "message": error_message,
                }
            )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# StaticFiles is mounted last so /api routes keep precedence.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
