"""ReAct-style tool-use loop: ask the model, execute any tool calls it makes, feed the
results back, repeat until it produces a final text answer or the step budget runs out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from sentinel.agent.llm import LLMClient
from sentinel.agent.tools import ToolSpec
from sentinel.config import settings
from sentinel.observability.tracing import traced_span

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Sentinel, an assistant that answers questions using the ingested knowledge base. "
    "Always call retrieve_documents before answering a factual question. If the retrieved "
    "context does not contain the answer, say so plainly instead of guessing."
)


@dataclass
class AgentStep:
    role: str
    content: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls_made: int = 0


def run_agent(
    query: str,
    llm: LLMClient,
    tools: list[ToolSpec],
    max_steps: int | None = None,
) -> AgentResult:
    max_steps = max_steps or settings.max_agent_steps
    tool_by_name = {t.name: t for t in tools}
    anthropic_tools = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]

    messages: list[dict] = [{"role": "user", "content": query}]
    steps: list[AgentStep] = [AgentStep(role="user", content=query)]
    tool_calls_made = 0

    with traced_span("agent.run", attributes={"query": query}):
        for step_num in range(max_steps):
            with traced_span("agent.step", attributes={"step": step_num}):
                response = llm.create_message(
                    messages=messages, tools=anthropic_tools, system=SYSTEM_PROMPT
                )

            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                steps.append(AgentStep(role="assistant", content=final_text))
                return AgentResult(answer=final_text, steps=steps, tool_calls_made=tool_calls_made)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool = tool_by_name.get(block.name)
                if tool is None:
                    output = f"error: unknown tool '{block.name}'"
                else:
                    with traced_span("agent.tool_call", attributes={"tool": block.name}):
                        output = tool.fn(**block.input)
                    tool_calls_made += 1
                steps.append(
                    AgentStep(role="tool", content=f"{block.name}({block.input}) -> {output}")
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            messages.append({"role": "user", "content": tool_results})

    logger.warning("agent_step_budget_exhausted", max_steps=max_steps)
    return AgentResult(
        answer="I wasn't able to reach a final answer within the step budget.",
        steps=steps,
        tool_calls_made=tool_calls_made,
    )
