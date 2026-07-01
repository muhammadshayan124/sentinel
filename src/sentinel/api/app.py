"""FastAPI service exposing ingestion + query over HTTP, for deploying Sentinel as a service."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from sentinel.agent.llm import AnthropicClient
from sentinel.agent.loop import run_agent
from sentinel.agent.tools import build_toolset
from sentinel.ingestion.pipeline import ingest_directory
from sentinel.observability.logging import configure_logging
from sentinel.observability.tracing import configure_tracing
from sentinel.retrieval.vector_store import VectorStore

configure_logging()
configure_tracing()

app = FastAPI(title="Sentinel", version="0.1.0")
_store = VectorStore()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    tool_calls_made: int


class IngestRequest(BaseModel):
    directory: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, int]:
    return _store.count()


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, int]:
    return ingest_directory(Path(req.directory), _store)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    llm = AnthropicClient()
    tools = build_toolset(_store)
    result = run_agent(req.question, llm=llm, tools=tools)
    return QueryResponse(answer=result.answer, tool_calls_made=result.tool_calls_made)
