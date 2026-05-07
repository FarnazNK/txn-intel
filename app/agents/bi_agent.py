"""Business intelligence agent.

Conversational tool-use loop over Claude. The agent answers questions about
customers, merchants, transactions, and support themes by calling SQL and
semantic-search tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agents.tools import TOOL_DEFINITIONS, TOOL_DISPATCH
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a business intelligence analyst for a multi-tenant transaction platform.

You have read-only access to the warehouse via tools. Your job:
1. Decompose the user's question into the tool calls that answer it.
2. Run those tools, inspect results, and iterate if needed.
3. Return a concise, data-grounded answer with the key numbers.

Guidelines:
- Prefer run_sql for aggregations, counts, sums, top-N lists.
- Use search_support_tickets when the question is about themes in customer feedback.
- Use get_customer_summary or get_merchant_summary for single-entity lookups.
- If a query returns nothing, try a broader filter or a different table.
- Never invent numbers. If the data doesn't support an answer, say so.
- Cite the figures you used. Round large numbers sensibly.
- Keep responses tight: lead with the answer, then 2-4 supporting numbers.
"""

MAX_TURNS = 8


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    turns: int = 0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _call_claude(client: anthropic.Anthropic, messages: list[dict]) -> Any:
    return client.messages.create(
        model=get_settings().anthropic_model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )


def ask(question: str, max_turns: int = MAX_TURNS) -> AgentResponse:
    settings = get_settings()
    if not settings.anthropic_api_key:
        return AgentResponse(answer="(agent disabled: ANTHROPIC_API_KEY not set)")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_calls_log: list[dict] = []
    final_text = ""

    for turn in range(max_turns):
        resp = _call_claude(client, messages)
        # Append assistant turn
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    log.info("turn %d: tool=%s input=%s", turn, tool_name, tool_input)
                    if tool_name not in TOOL_DISPATCH:
                        result = {"error": f"unknown tool {tool_name}"}
                    else:
                        try:
                            result = TOOL_DISPATCH[tool_name](tool_input)
                        except Exception as e:
                            result = {"error": f"{type(e).__name__}: {e}"}
                    tool_calls_log.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result_summary": _summarize_result(result),
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)[:8000],
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Stop reason is end_turn or similar — extract final text
        for block in resp.content:
            if block.type == "text":
                final_text += block.text
        return AgentResponse(answer=final_text.strip(), tool_calls=tool_calls_log, turns=turn + 1)

    return AgentResponse(
        answer="(agent hit max turns without converging) " + final_text,
        tool_calls=tool_calls_log,
        turns=max_turns,
    )


def _summarize_result(result: dict) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    if "rows" in result:
        return f"{result.get('row_count', len(result['rows']))} rows"
    if "results" in result:
        return f"{len(result['results'])} hits"
    return "ok"
