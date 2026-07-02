"""Typer CLI: `sentinel ingest`, `sentinel query`, `sentinel eval`, `sentinel serve`."""

from __future__ import annotations

from pathlib import Path

import typer

from sentinel.agent.llm import get_llm_client
from sentinel.agent.loop import run_agent
from sentinel.agent.tools import build_toolset
from sentinel.config import settings
from sentinel.ingestion.pipeline import ingest_directory
from sentinel.observability.logging import configure_logging
from sentinel.observability.tracing import configure_tracing
from sentinel.retrieval.vector_store import VectorStore

app = typer.Typer(help="Sentinel: autonomous multimodal RAG agent")


@app.callback()
def _init() -> None:
    configure_logging()
    configure_tracing()


@app.command()
def ingest(directory: Path = typer.Argument(settings.data_dir)) -> None:
    """Ingest a directory of PDFs/text/images into the vector store."""
    store = VectorStore()
    summary = ingest_directory(directory, store)
    typer.echo(summary)


@app.command()
def query(question: str) -> None:
    """Ask the agent a question against the ingested knowledge base."""
    store = VectorStore()
    llm = get_llm_client()
    tools = build_toolset(store)
    result = run_agent(question, llm=llm, tools=tools)
    typer.echo(result.answer)


@app.command()
def eval(
    dataset: Path = typer.Argument(..., help="Path to eval dataset JSON"),
    report: Path = typer.Option(None, help="Where to write the JSON report"),
) -> None:
    """Run the eval suite against the current vector store."""
    from sentinel.eval.runner import run_eval

    store = VectorStore()
    result = run_eval(dataset, store, report_path=report)
    typer.echo(f"retrieval_recall={result.retrieval_recall:.2f}")
    typer.echo(f"mean_keyword_coverage={result.mean_keyword_coverage:.2f}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("sentinel.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
