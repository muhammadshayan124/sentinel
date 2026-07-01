"""Tool definitions the agent can call. Each tool is a plain function plus an Anthropic
tool-use schema, kept together so they can never drift out of sync.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sentinel.retrieval.retriever import retrieve
from sentinel.retrieval.vector_store import VectorStore

_ALLOWED_OPERATORS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (no eval/exec, AST-walked whitelist)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:
        return f"error: could not evaluate '{expression}': {exc}"


def make_retrieve_tool(store: VectorStore) -> Callable[[str], str]:
    def retrieve_documents(query: str) -> str:
        result = retrieve(query, store)
        return result.as_context_string()

    return retrieve_documents


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]


def build_toolset(store: VectorStore) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="retrieve_documents",
            description=(
                "Search the ingested knowledge base (text chunks and images) for content "
                "relevant to a query. Use this before answering any factual question about "
                "the ingested corpus."
            ),
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            fn=make_retrieve_tool(store),
        ),
        ToolSpec(
            name="calculator",
            description="Evaluate a basic arithmetic expression, e.g. '(3 + 4) * 2'.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression"}
                },
                "required": ["expression"],
            },
            fn=calculator,
        ),
    ]
