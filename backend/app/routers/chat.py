"""Chat endpoint — answers questions about a specific ticker using OpenAI.

Builds a context window from DB data (prices, signals, CIO decision, agent
verdicts) and streams the response back via SSE.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_session
from app.db.models import (
    AgentVerdictRecord,
    CIODecisionRecord,
    Fundamentals,
    NewsItem,
    PriceOHLCV,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

SYSTEM_PROMPT = """You are a quantitative analyst assistant for a hedge fund AI trading system.
You have access to real market data, agent verdicts, and CIO decisions for the ticker being discussed.
Answer questions concisely and factually based on the provided context.
Use numbers when available. Be direct — no fluff.
If data is missing, say so clearly rather than guessing."""


class ChatRequest(BaseModel):
    ticker: str
    message: str
    opportunity_id: str | None = None


async def _build_context(ticker: str, opportunity_id: str | None, session: AsyncSession) -> str:
    """Pull relevant DB data and format as context for the LLM."""
    parts: list[str] = [f"=== CONTEXT FOR {ticker.upper()} ===\n"]

    # Latest price
    price_stmt = (
        select(PriceOHLCV)
        .where(PriceOHLCV.ticker == ticker.upper())
        .order_by(desc(PriceOHLCV.timestamp))
        .limit(5)
    )
    price_rows = list((await session.execute(price_stmt)).scalars())
    if price_rows:
        latest = price_rows[0]
        parts.append(
            f"PRICE: ${latest.close:.2f} | "
            f"H {latest.high:.2f} L {latest.low:.2f} | "
            f"Vol {int(latest.volume or 0):,} | "
            f"Date {latest.timestamp.date()}"
        )
        if len(price_rows) > 1:
            oldest = price_rows[-1]
            chg = ((latest.close - oldest.close) / oldest.close) * 100
            parts.append(f"5-day change: {chg:+.1f}%")

    # Fundamentals
    try:
        fund_stmt = (
            select(Fundamentals)
            .where(Fundamentals.ticker == ticker.upper())
            .order_by(desc(Fundamentals.timestamp))
            .limit(1)
        )
        fund_row = (await session.execute(fund_stmt)).scalars().first()
        if fund_row:
            parts.append(
                f"FUNDAMENTALS: Market cap ${(fund_row.market_cap or 0)/1e6:.0f}M | "
                f"P/E {fund_row.pe_ratio or 'N/A'} | "
                f"Revenue ${(fund_row.revenue or 0)/1e6:.0f}M | "
                f"FCF ${(fund_row.free_cash_flow or 0)/1e6:.0f}M"
            )
    except Exception:
        pass

    # Latest news headlines
    try:
        news_stmt = (
            select(NewsItem)
            .where(NewsItem.ticker == ticker.upper())
            .order_by(desc(NewsItem.published_at))
            .limit(5)
        )
        news_rows = list((await session.execute(news_stmt)).scalars())
        if news_rows:
            headlines = [f"  - {n.headline}" for n in news_rows]
            parts.append("RECENT NEWS:\n" + "\n".join(headlines))
    except Exception:
        pass

    # Latest CIO decision
    try:
        cio_stmt = (
            select(CIODecisionRecord)
            .order_by(desc(CIODecisionRecord.decided_at))
            .limit(20)
        )
        cio_rows = list((await session.execute(cio_stmt)).scalars())
        # Find matching ticker
        for row in cio_rows:
            try:
                d = json.loads(row.decision_json)
                if d.get("ticker", "").upper() == ticker.upper() or opportunity_id == row.opportunity_id:
                    parts.append(
                        f"CIO DECISION: {d.get('decision', row.final_verdict)} | "
                        f"Score {d.get('weighted_score', row.conviction_score):.0f}/100 | "
                        f"Size {d.get('position_size_pct', row.suggested_allocation_pct):.1f}%"
                    )
                    # Monte Carlo
                    mc = d.get("monte_carlo", {})
                    if mc:
                        parts.append(
                            f"MONTE CARLO: P(profit) {mc.get('prob_profit', 0)*100:.0f}% | "
                            f"VaR95 {mc.get('var_95', 0):.1f}% | "
                            f"Expected {mc.get('expected_value', 0):.2f}"
                        )
                    break
            except Exception:
                continue
    except Exception:
        pass

    # Agent verdicts
    try:
        opp_id = opportunity_id
        if not opp_id:
            # Find most recent for ticker
            all_cio = list((await session.execute(
                select(CIODecisionRecord).order_by(desc(CIODecisionRecord.decided_at)).limit(50)
            )).scalars())
            for row in all_cio:
                try:
                    d = json.loads(row.decision_json)
                    if d.get("ticker", "").upper() == ticker.upper():
                        opp_id = row.opportunity_id
                        break
                except Exception:
                    continue

        if opp_id:
            verdict_stmt = (
                select(AgentVerdictRecord)
                .where(AgentVerdictRecord.opportunity_id == opp_id)
            )
            verdict_rows = list((await session.execute(verdict_stmt)).scalars())
            if verdict_rows:
                agent_lines = []
                for vrow in verdict_rows:
                    try:
                        v = json.loads(vrow.verdict_json)
                        agent_lines.append(
                            f"  {v.get('agent_id', vrow.persona).upper()}: "
                            f"{v.get('direction', vrow.verdict)} | "
                            f"score {v.get('score', vrow.confidence)}/100 | "
                            f"bull: {', '.join(v.get('bull_factors', [])[:2])} | "
                            f"bear: {', '.join(v.get('bear_factors', [])[:2])}"
                        )
                    except Exception:
                        agent_lines.append(f"  {vrow.persona}: {vrow.verdict} ({vrow.confidence})")
                parts.append("AGENT VERDICTS:\n" + "\n".join(agent_lines))
    except Exception:
        pass

    return "\n".join(parts)


async def _stream_chat(context: str, message: str) -> AsyncGenerator[str, None]:
    """Stream OpenAI response as SSE chunks."""
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQUESTION: {message}"},
        ],
        stream=True,
        max_tokens=600,
        temperature=0.3,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {json.dumps({'text': delta})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("")
async def chat(
    req: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Stream an AI answer about a ticker based on live DB context."""
    context = await _build_context(req.ticker, req.opportunity_id, session)
    logger.info("Chat request: ticker=%s msg=%s", req.ticker, req.message[:80])

    return StreamingResponse(
        _stream_chat(context, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
