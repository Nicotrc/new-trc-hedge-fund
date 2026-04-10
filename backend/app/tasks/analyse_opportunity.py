"""Agent fan-out pipeline V2: BLPOP consumer, quant agent fan-out, CIO v2.

Architecture (V2):
  consume_queue       -- Long-running BLPOP consumer; dispatches fan_out_v2.
  fan_out_v2          -- Dispatches 5 run_quant_agent tasks in parallel.
  run_quant_agent     -- Invokes QUANT_AGENT_GRAPH for a single agent,
                         persists QuantAgentVerdict to Redis, increments counter.
  run_committee_v2    -- Triggered when counter == 5; runs Meta-Agent + CIO v2
                         + paper trade execution + SSE events + DB persist.

V1 tasks (run_persona_agent, run_committee, fan_out) are preserved for
backward compatibility but are no longer called by the primary pipeline.

Fan-in: Redis HINCRBY atomic counter (unchanged from V1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import redis

from app.agents.quant_agents.graph import QUANT_AGENT_GRAPH
from app.agents.schemas import QuantAgentVerdict
from app.events.publisher import publish_event
from app.signals.queue import OPP_QUEUE_KEY
from app.tasks.celery_app import app

logger = logging.getLogger(__name__)

_REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

QUANT_AGENTS = ["momentum", "value", "event", "macro", "risk"]
AGENT_COUNT = len(QUANT_AGENTS)
VERDICT_TTL = 86400  # 24 hours


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_quant_graph_sync(
    agent_id: str,
    data_context: dict[str, Any],
    redis_url: str,
) -> dict[str, Any]:
    """Invoke QUANT_AGENT_GRAPH synchronously from a Celery worker."""

    async def _invoke() -> dict[str, Any]:
        result = await QUANT_AGENT_GRAPH.ainvoke({
            "agent_id": agent_id,
            "data_context": data_context,
            "redis_url": redis_url,
            "verdict": None,
        })
        return result["verdict"].model_dump()

    try:
        return asyncio.run(_invoke())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(_invoke())


def _build_v2_data_context(opportunity: dict) -> dict[str, Any]:
    """Build the full 8-category data context for V2 quant agents.

    Queries all data tables and assembles a unified context dict.
    The DataPartitioner will then filter per-agent.
    """
    from datetime import timedelta

    from sqlalchemy import select

    from app.db.engine import SyncSessionLocal
    from app.db.models import (
        EventCalendar,
        Fundamentals,
        InsiderTrade,
        MacroIndicator,
        NewsItem,
        OptionsFlow,
        PriceOHLCV,
        ShortInterest,
    )

    ticker = opportunity.get("ticker", "")
    now = datetime.now(timezone.utc)

    context: dict[str, Any] = {
        "ticker": ticker,
        "composite_score": opportunity.get("composite_score"),
        "detected_at": opportunity.get("detected_at"),
        "strategy": opportunity.get("strategy"),
        "binary_event": opportunity.get("binary_event", False),
        "price_action": [],
        "fundamentals": [],
        "insider_trades": [],
        "news": [],
        "macro_indicators": [],
        "options_flow": [],
        "event_calendar": [],
        "short_interest": [],
    }

    with SyncSessionLocal() as session:
        # price_action — last 60 days
        price_rows = session.execute(
            select(PriceOHLCV)
            .where(PriceOHLCV.ticker == ticker)
            .where(PriceOHLCV.timestamp >= now - timedelta(days=65))
            .order_by(PriceOHLCV.timestamp.asc())
            .limit(65)
        ).scalars().all()
        for p in price_rows:
            context["price_action"].append({
                "date": p.timestamp.isoformat(),
                "open": float(p.open) if p.open else None,
                "high": float(p.high) if p.high else None,
                "low": float(p.low) if p.low else None,
                "close": float(p.close) if p.close else None,
                "volume": p.volume,
            })

        # fundamentals — latest row
        fund_row = session.execute(
            select(Fundamentals)
            .where(Fundamentals.ticker == ticker)
            .order_by(Fundamentals.timestamp.desc())
            .limit(1)
        ).scalars().first()
        if fund_row:
            context["fundamentals"].append({
                "pe_ratio": float(fund_row.pe_ratio) if fund_row.pe_ratio else None,
                "revenue": float(fund_row.revenue) if fund_row.revenue else None,
                "net_income": float(fund_row.net_income) if fund_row.net_income else None,
                "eps": float(fund_row.eps) if fund_row.eps else None,
                "debt_to_equity": float(fund_row.debt_to_equity) if fund_row.debt_to_equity else None,
                "free_cash_flow": float(fund_row.free_cash_flow) if fund_row.free_cash_flow else None,
                "market_cap": float(fund_row.market_cap) if fund_row.market_cap else None,
                "as_of": fund_row.timestamp.isoformat(),
            })

        # insider_trades — last 90 days
        insider_rows = session.execute(
            select(InsiderTrade)
            .where(InsiderTrade.ticker == ticker)
            .where(InsiderTrade.timestamp >= now - timedelta(days=90))
            .order_by(InsiderTrade.timestamp.desc())
            .limit(20)
        ).scalars().all()
        for i in insider_rows:
            context["insider_trades"].append({
                "insider_name": i.insider_name,
                "trade_type": i.trade_type,
                "shares": i.shares,
                "value": float(i.trade_value) if i.trade_value else None,
                "date": i.timestamp.isoformat(),
            })

        # news — last 7 days
        news_rows = session.execute(
            select(NewsItem)
            .where(NewsItem.ticker == ticker)
            .where(NewsItem.timestamp >= now - timedelta(days=7))
            .order_by(NewsItem.timestamp.desc())
            .limit(15)
        ).scalars().all()
        for n in news_rows:
            context["news"].append({
                "headline": n.headline,
                "summary": n.summary,
                "sentiment": n.sentiment,
                "published": n.timestamp.isoformat(),
            })

        # macro_indicators — last 7 days
        macro_rows = session.execute(
            select(MacroIndicator)
            .where(MacroIndicator.timestamp >= now - timedelta(days=7))
            .order_by(MacroIndicator.timestamp.desc())
            .limit(30)
        ).scalars().all()
        for m in macro_rows:
            context["macro_indicators"].append({
                "indicator": m.indicator,
                "value": float(m.value) if m.value else None,
                "change_1d": float(m.change_1d) if m.change_1d else None,
                "change_5d": float(m.change_5d) if m.change_5d else None,
                "as_of": m.timestamp.isoformat(),
            })

        # options_flow — latest
        opt_row = session.execute(
            select(OptionsFlow)
            .where(OptionsFlow.ticker == ticker)
            .order_by(OptionsFlow.timestamp.desc())
            .limit(1)
        ).scalars().first()
        if opt_row:
            context["options_flow"].append({
                "call_volume": opt_row.call_volume,
                "put_volume": opt_row.put_volume,
                "put_call_ratio": float(opt_row.put_call_ratio) if opt_row.put_call_ratio else None,
                "unusual_activity_score": float(opt_row.unusual_activity_score) if opt_row.unusual_activity_score else None,
                "iv_rank": float(opt_row.iv_rank) if opt_row.iv_rank else None,
                "as_of": opt_row.timestamp.isoformat(),
            })

        # event_calendar — upcoming 45 days
        event_rows = session.execute(
            select(EventCalendar)
            .where(EventCalendar.ticker == ticker)
            .where(EventCalendar.event_date >= now)
            .where(EventCalendar.event_date <= now + timedelta(days=45))
            .order_by(EventCalendar.event_date.asc())
            .limit(5)
        ).scalars().all()
        for e in event_rows:
            context["event_calendar"].append({
                "event_type": e.event_type,
                "event_date": e.event_date.isoformat() if e.event_date else None,
                "description": e.description,
                "impact_estimate": e.impact_estimate,
                "binary_outcome": e.binary_outcome,
            })

        # short_interest — latest
        si_row = session.execute(
            select(ShortInterest)
            .where(ShortInterest.ticker == ticker)
            .order_by(ShortInterest.timestamp.desc())
            .limit(1)
        ).scalars().first()
        if si_row:
            context["short_interest"].append({
                "si_pct": float(si_row.si_pct) if si_row.si_pct else None,
                "days_to_cover": float(si_row.days_to_cover) if si_row.days_to_cover else None,
                "borrow_rate": float(si_row.borrow_rate) if si_row.borrow_rate else None,
                "shares_short": si_row.shares_short,
                "as_of": si_row.timestamp.isoformat(),
            })

    return context


# ---------------------------------------------------------------------------
# V2 Celery tasks
# ---------------------------------------------------------------------------


@app.task(
    name="app.tasks.analyse_opportunity.run_quant_agent",
    bind=True,
    max_retries=3,
)
def run_quant_agent(
    self,
    opportunity_id: str,
    agent_id: str,
    data_context: dict,
) -> None:
    """Invoke a single V2 quant agent, persist verdict to Redis, increment counter.

    Publishes AGENT_STARTED before and AGENT_COMPLETE after the LLM call.
    When all AGENT_COUNT verdicts are in, dispatches run_committee_v2.
    """
    r = redis.from_url(_REDIS_URL)
    try:
        ticker = opportunity_id.split(":", 1)[0] if ":" in opportunity_id else opportunity_id

        publish_event(r, "AGENT_STARTED", {
            "opportunity_id": opportunity_id,
            "agent": agent_id,
            "ticker": ticker,
            "version": "v2",
        })

        try:
            verdict_dict = _run_quant_graph_sync(agent_id, data_context, _REDIS_URL)
        except Exception as exc:
            logger.exception("run_quant_agent failed for %s/%s — retrying", opportunity_id, agent_id)
            raise self.retry(exc=exc, countdown=30)

        r.hset(f"verdicts_v2:{opportunity_id}", agent_id, json.dumps(verdict_dict, default=str))
        r.expire(f"verdicts_v2:{opportunity_id}", VERDICT_TTL)

        completed = int(r.hincrby(f"verdicts_counter_v2:{opportunity_id}", "count", 1))
        r.expire(f"verdicts_counter_v2:{opportunity_id}", VERDICT_TTL)

        publish_event(r, "AGENT_COMPLETE", {
            "opportunity_id": opportunity_id,
            "agent": agent_id,
            "ticker": ticker,
            "score": verdict_dict.get("score"),
            "direction": verdict_dict.get("direction"),
            "conviction": verdict_dict.get("conviction"),
            "version": "v2",
        })

        logger.info(
            "Quant agent %s complete for %s — %d/%d done",
            agent_id, opportunity_id, completed, AGENT_COUNT,
        )

        if completed >= AGENT_COUNT:
            logger.info("All V2 agents complete for %s — triggering committee_v2", opportunity_id)
            run_committee_v2.delay(opportunity_id)

    finally:
        r.close()


@app.task(
    name="app.tasks.analyse_opportunity.fan_out_v2",
    bind=True,
)
def fan_out_v2(self, opportunity: dict) -> None:
    """Dispatch one run_quant_agent task per V2 agent in parallel.

    Builds the 8-category data context, stores it in Redis,
    then dispatches AGENT_COUNT (5) tasks.
    """
    opportunity_id = f"{opportunity['ticker']}:{opportunity['detected_at']}"
    data_context = _build_v2_data_context(opportunity)

    r = redis.from_url(_REDIS_URL)
    try:
        r.set(
            f"opportunity:{opportunity_id}",
            json.dumps(opportunity, default=str),
            ex=VERDICT_TTL,
        )
    finally:
        r.close()

    logger.info("V2 fan-out to %d quant agents for %s", AGENT_COUNT, opportunity_id)

    for agent_id in QUANT_AGENTS:
        run_quant_agent.delay(opportunity_id, agent_id, data_context)


@app.task(
    name="app.tasks.analyse_opportunity.run_committee_v2",
    bind=True,
)
def run_committee_v2(self, opportunity_id: str) -> None:
    """Aggregate V2 agent verdicts → Meta-Agent → CIO v2 → paper trade → DB.

    Full pipeline:
      1. Load QuantAgentVerdict objects from Redis
      2. Load opportunity from Redis
      3. Get current price for MC simulation
      4. Load agent accuracy from Redis (Learning System)
      5. make_cio_decision_v2() — MC + Kelly + risk adjustment
      6. Publish COMMITTEE_COMPLETE + DECISION_MADE events
      7. Execute paper trade if BUY/SELL
      8. Persist to DB (AgentVerdictRecord + CIODecisionRecord)
      9. Update agent weights (Learning System)
      10. Cleanup Redis keys
    """
    from app.analysis.cio_v2 import make_cio_decision_v2
    from app.db.engine import SyncSessionLocal
    from app.db.models import AgentVerdictRecord, CIODecisionRecord
    from app.execution.paper_trading import PaperTradingEngine
    from app.execution.position_manager import PositionManager
    from app.learning.weight_adjuster import get_current_accuracies

    r = redis.from_url(_REDIS_URL)
    try:
        # --- 1. Load verdicts ---
        raw_verdicts = r.hgetall(f"verdicts_v2:{opportunity_id}")
        verdicts: list[QuantAgentVerdict] = []
        for agent_id, raw in raw_verdicts.items():
            agent_str = agent_id.decode() if isinstance(agent_id, bytes) else agent_id
            raw_str = raw.decode() if isinstance(raw, bytes) else raw
            try:
                d = json.loads(raw_str)
                # Reconstruct KeyLevels nested model
                from app.agents.schemas import KeyLevels
                if isinstance(d.get("key_levels"), dict):
                    d["key_levels"] = KeyLevels(**d["key_levels"])
                verdicts.append(QuantAgentVerdict(**d))
            except Exception:
                logger.exception("Failed to parse verdict for agent %s", agent_str)

        if not verdicts:
            logger.error("No verdicts found for %s — aborting committee_v2", opportunity_id)
            return

        # --- 2. Load opportunity ---
        raw_opp = r.get(f"opportunity:{opportunity_id}")
        opportunity = json.loads(raw_opp) if raw_opp else {}
        ticker = opportunity_id.split(":", 1)[0] if ":" in opportunity_id else opportunity_id

        # --- 3. Estimate current price from price_action in opportunity context ---
        current_price = _get_current_price(ticker, r)

        # --- 4. Load agent accuracy from Redis (Learning System) ---
        accuracy_lookup = get_current_accuracies(r)

        # --- 5. CIO v2 decision (includes Meta-Agent + MC) ---
        decision = make_cio_decision_v2(
            opportunity_id=opportunity_id,
            ticker=ticker,
            verdicts=verdicts,
            current_price=current_price,
            realized_vol_20d=0.45,  # default; TODO: compute from price history
            accuracy_lookup=accuracy_lookup,
        )

        # --- 6. Publish events ---
        publish_event(r, "COMMITTEE_COMPLETE", {
            "opportunity_id": opportunity_id,
            "ticker": ticker,
            "weighted_score": decision.weighted_score,
            "version": "v2",
            "agent_scores": {v.agent_id: v.score for v in verdicts},
        })

        publish_event(r, "DECISION_MADE", {
            "opportunity_id": opportunity_id,
            "ticker": ticker,
            "decision": {
                "decision": decision.decision,
                "weighted_score": decision.weighted_score,
                "position_size_pct": decision.position_size_pct,
                "risk_reward_ratio": decision.risk_reward_ratio,
                "entry_price": decision.entry_price,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "prob_profit": decision.monte_carlo.prob_profit,
                "veto_triggered": decision.veto_triggered,
                "veto_reason": decision.veto_reason,
                "kelly_fraction": decision.kelly_fraction,
                "binary_event": any(v.binary_event for v in verdicts),
            },
            "version": "v2",
        })

        if decision.veto_triggered:
            publish_event(r, "RISK_VETO", {
                "opportunity_id": opportunity_id,
                "ticker": ticker,
                "reason": decision.veto_reason,
                "risk_agent_score": decision.risk_agent_score,
            })

        # --- 7. Execute paper trade ---
        if decision.decision in ("BUY", "SELL") and current_price > 0:
            try:
                pm = PositionManager(r)
                if not pm.has_position(ticker):
                    _execute_paper_trade(decision, current_price, pm)
            except Exception:
                logger.exception("Paper trade execution failed for %s", opportunity_id)

        # --- 8. Persist to DB ---
        now = datetime.now(timezone.utc)
        with SyncSessionLocal() as session:
            for v in verdicts:
                record = AgentVerdictRecord(
                    analysed_at=now,
                    opportunity_id=opportunity_id,
                    persona=v.agent_id,
                    verdict=v.direction,
                    confidence=int(v.confidence * 100),
                    verdict_json=json.dumps(v.model_dump(), default=str),
                )
                session.merge(record)

            cio_record = CIODecisionRecord(
                decided_at=now,
                opportunity_id=opportunity_id,
                conviction_score=int(decision.weighted_score),
                suggested_allocation_pct=decision.position_size_pct,
                final_verdict=decision.decision,
                decision_json=json.dumps(decision.model_dump(), default=str),
            )
            session.merge(cio_record)
            session.commit()

        logger.info(
            "V2 committee complete for %s — decision=%s score=%.1f size=%.1f%% veto=%s",
            opportunity_id,
            decision.decision,
            decision.weighted_score,
            decision.position_size_pct,
            decision.veto_triggered,
        )

        # --- 9. Update agent weights ---
        try:
            from app.learning.weight_adjuster import update_agent_weights
            update_agent_weights(r)
        except Exception:
            logger.debug("Weight adjuster skipped (insufficient trade history)")

        # --- 10. Cleanup ---
        r.delete(
            f"verdicts_v2:{opportunity_id}",
            f"verdicts_counter_v2:{opportunity_id}",
            f"opportunity:{opportunity_id}",
        )

    finally:
        r.close()


def _get_current_price(ticker: str, r: Any) -> float:
    """Estimate current price from DB or YFinance fallback."""
    try:
        from sqlalchemy import select
        from app.db.engine import SyncSessionLocal
        from app.db.models import PriceOHLCV
        with SyncSessionLocal() as session:
            row = session.execute(
                select(PriceOHLCV)
                .where(PriceOHLCV.ticker == ticker)
                .order_by(PriceOHLCV.timestamp.desc())
                .limit(1)
            ).scalars().first()
            if row and row.close:
                return float(row.close)
    except Exception:
        pass

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
        if price and float(price) > 0:
            return float(price)
    except Exception:
        pass

    logger.warning("Could not determine current price for %s — defaulting to 100.0", ticker)
    return 100.0


def _execute_paper_trade(decision, current_price: float, pm) -> None:
    """Execute a paper trade via PositionManager."""
    from app.agents.schemas import Position
    import uuid

    position_value = 100_000.0 * (decision.position_size_pct / 100.0)
    shares = max(1, int(position_value / current_price))

    # Risk-per-trade validation
    if decision.stop_loss and current_price > 0:
        risk_per_share = abs(current_price - decision.stop_loss)
        max_risk_dollars = 100_000.0 * 0.02
        if risk_per_share > 0 and risk_per_share * shares > max_risk_dollars:
            shares = max(1, int(max_risk_dollars / risk_per_share))

    position = Position(
        position_id=str(uuid.uuid4()),
        opportunity_id=decision.opportunity_id,
        ticker=decision.ticker,
        entry_price=current_price,
        shares=shares,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        entry_date=datetime.now(timezone.utc).isoformat(),
        decision_id=decision.opportunity_id,
    )
    pm.save_position(position)
    pm.adjust_capital(-(current_price * shares))
    logger.info(
        "Paper trade opened: %s %d shares @ $%.2f SL=$%s TP=$%s",
        decision.ticker,
        shares,
        current_price,
        f"{decision.stop_loss:.2f}" if decision.stop_loss else "none",
        f"{decision.take_profit:.2f}" if decision.take_profit else "none",
    )


# ---------------------------------------------------------------------------
# Main queue consumer
# ---------------------------------------------------------------------------


@app.task(
    name="app.tasks.analyse_opportunity.consume_queue",
    bind=True,
)
def consume_queue(self) -> None:
    """Long-running BLPOP consumer for the opportunity queue (V2).

    Routes each dequeued opportunity to fan_out_v2.
    """
    r = redis.from_url(_REDIS_URL)
    logger.info("consume_queue (V2) started — listening on %s", OPP_QUEUE_KEY)

    while True:
        result = r.blpop(OPP_QUEUE_KEY, timeout=30)
        if result is None:
            continue

        _, raw = result
        opportunity = json.loads(raw)
        logger.info(
            "Dequeued opportunity: %s (detected_at=%s strategy=%s)",
            opportunity.get("ticker"),
            opportunity.get("detected_at"),
            opportunity.get("strategy"),
        )
        fan_out_v2.delay(opportunity)


# ---------------------------------------------------------------------------
# V1 tasks — kept for backward compatibility (not called by V2 pipeline)
# ---------------------------------------------------------------------------


@app.task(
    name="app.tasks.analyse_opportunity.run_persona_agent",
    bind=True,
    max_retries=3,
)
def run_persona_agent(self, opportunity_id: str, persona_name: str, data_context: dict) -> None:
    """[V1 LEGACY] Run a persona-based agent. Kept for backward compatibility."""
    from app.agents.graph import PERSONA_GRAPH
    from app.agents.schemas import AgentVerdict

    r = redis.from_url(_REDIS_URL)
    try:
        ticker = opportunity_id.split(":", 1)[0] if ":" in opportunity_id else opportunity_id

        async def _invoke():
            result = await PERSONA_GRAPH.ainvoke({
                "persona_name": persona_name,
                "data_context": data_context,
                "redis_url": _REDIS_URL,
                "verdict": None,
            })
            return result["verdict"].model_dump()

        try:
            verdict_dict = asyncio.run(_invoke())
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            verdict_dict = asyncio.run(_invoke())

        r.hset(f"verdicts:{opportunity_id}", persona_name, json.dumps(verdict_dict, default=str))
        r.expire(f"verdicts:{opportunity_id}", VERDICT_TTL)
        completed = int(r.hincrby(f"verdicts_counter:{opportunity_id}", "count", 1))
        r.expire(f"verdicts_counter:{opportunity_id}", VERDICT_TTL)

        if completed >= 5:
            run_committee.delay(opportunity_id)

    finally:
        r.close()


@app.task(
    name="app.tasks.analyse_opportunity.fan_out",
    bind=True,
)
def fan_out(self, opportunity: dict) -> None:
    """[V1 LEGACY] Fan out to persona agents."""
    opportunity_id = f"{opportunity['ticker']}:{opportunity['detected_at']}"
    r = redis.from_url(_REDIS_URL)
    try:
        r.set(f"opportunity:{opportunity_id}", json.dumps(opportunity, default=str), ex=VERDICT_TTL)
    finally:
        r.close()
    # Route to V2 instead
    fan_out_v2.delay(opportunity)


@app.task(name="app.tasks.analyse_opportunity.run_committee", bind=True)
def run_committee(self, opportunity_id: str) -> None:
    """[V1 LEGACY] Persona-based committee. Kept for backward compat."""
    logger.info("V1 run_committee called for %s — routing to V2", opportunity_id)
    run_committee_v2.delay(opportunity_id)
