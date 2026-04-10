"""LangGraph graph for V2 quant agent invocation.

Routes to the correct quant agent (momentum/value/event/macro/risk) based on
the agent_id in state. Compiled once at import time.

Graph topology: START -> run_quant_agent -> END
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.schemas import QuantAgentVerdict


class QuantAgentState(TypedDict):
    """State passed through the quant agent graph.

    agent_id: One of {"momentum", "value", "event", "macro", "risk"}.
    data_context: Full financial data dict — partitioner filters per-agent.
    redis_url: Redis connection URL string.
    verdict: Populated by run_quant_agent_node; None on entry.
    """

    agent_id: str
    data_context: dict[str, Any]
    redis_url: str
    verdict: Optional[QuantAgentVerdict]


async def run_quant_agent_node(state: QuantAgentState) -> dict[str, Any]:
    """LangGraph node: route to the correct quant agent and return the verdict.

    Deferred imports avoid circular imports at module-load time.
    """
    import redis.asyncio as aioredis

    agent_id = state["agent_id"]
    data_context = state["data_context"]
    redis_url = state["redis_url"]

    redis_client = aioredis.from_url(redis_url)
    try:
        if agent_id == "momentum":
            from app.agents.quant_agents.momentum_agent import run_momentum_agent
            verdict = await run_momentum_agent(data_context, redis_client)
        elif agent_id == "value":
            from app.agents.quant_agents.value_agent import run_value_agent
            verdict = await run_value_agent(data_context, redis_client)
        elif agent_id == "event":
            from app.agents.quant_agents.event_agent import run_event_agent
            verdict = await run_event_agent(data_context, redis_client)
        elif agent_id == "macro":
            from app.agents.quant_agents.macro_agent import run_macro_agent
            verdict = await run_macro_agent(data_context, redis_client)
        elif agent_id == "risk":
            from app.agents.quant_agents.risk_agent import run_risk_agent
            verdict = await run_risk_agent(data_context, redis_client)
        else:
            raise ValueError(f"Unknown quant agent id: '{agent_id}'")
    finally:
        await redis_client.aclose()

    return {"verdict": verdict}


def build_quant_agent_graph():
    """Construct and compile the quant agent StateGraph.

    Returns:
        A compiled LangGraph CompiledGraph that accepts QuantAgentState and
        returns a dict containing the populated ``verdict`` field.
    """
    builder = StateGraph(QuantAgentState)
    builder.add_node("run_quant_agent", run_quant_agent_node)
    builder.add_edge(START, "run_quant_agent")
    builder.add_edge("run_quant_agent", END)
    return builder.compile()


# Compiled once at import time — safe to import from Celery tasks.
QUANT_AGENT_GRAPH = build_quant_agent_graph()
