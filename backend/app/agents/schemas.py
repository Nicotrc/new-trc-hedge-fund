"""Pydantic schemas for the agent analysis pipeline.

V1 schemas (AgentVerdict, CommitteeReport, CIODecision) are retained for
backward compatibility with existing committee/CIO logic.

V2 schemas (QuantAgentVerdict, MetaAgentReport, CIODecisionV2, MonteCarloResult)
implement the quantitative redesign — every field has an exact numerical consumer
downstream. No narrative bottlenecks.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# V1 — Legacy schemas (kept for committee.py / cio.py backward compat)
# ===========================================================================


class AgentVerdict(BaseModel):
    """Per-persona investment verdict (V1 — narrative-based)."""

    persona: str
    verdict: Literal["BUY", "HOLD", "PASS"]
    confidence: int = Field(ge=0, le=100)
    rationale: str
    key_metrics_used: List[str]
    risks: List[str]
    upside_scenario: str
    time_horizon: str
    data_gaps: List[str]


class CommitteeReport(BaseModel):
    """Aggregated report from all five persona agents (V1)."""

    opportunity_id: str
    verdicts: List[AgentVerdict]
    consensus: Literal["BUY", "HOLD", "PASS", "SPLIT"]
    dissent_agents: List[str]
    variance_score: float
    weighted_conviction: float
    asymmetric_flag: bool
    asymmetric_justification: Optional[str] = None


class CIODecision(BaseModel):
    """CIO-level final decision (V1 — deterministic rules)."""

    opportunity_id: str
    conviction_score: int = Field(ge=0, le=100)
    suggested_allocation_pct: float = Field(ge=0.0, le=100.0)
    time_horizon: str
    risk_rating: Literal["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    key_catalysts: List[str]
    kill_conditions: List[str]
    final_verdict: Literal["INVEST", "MONITOR", "PASS"]


# ===========================================================================
# V2 — Quantitative schemas
# ===========================================================================


class KeyLevels(BaseModel):
    """Price levels for trade execution."""

    entry: Optional[float] = None
    stop: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None


class QuantAgentVerdict(BaseModel):
    """Per-agent quantitative verdict (V2).

    Every field has an exact consumer downstream.
    No narrative fields — structured factors only.
    """

    agent_id: str  # "momentum" | "value" | "event" | "macro" | "risk"
    score: int = Field(ge=0, le=100)  # model-derived quality score
    expected_return_pct: float  # feeds MC drift parameter
    max_loss_pct: float  # feeds VaR/stop calibration
    risk_reward_ratio: float  # expected_return / max_loss
    confidence: float = Field(ge=0.0, le=1.0)  # data quality factor
    direction: Literal["LONG", "SHORT", "FLAT"]
    conviction: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    bull_factors: List[str] = Field(max_length=3)  # structured, max 3
    bear_factors: List[str] = Field(max_length=3)  # structured, max 3
    key_levels: KeyLevels
    time_horizon_days: int  # feeds MC simulation length
    catalyst_date: Optional[str] = None  # ISO date — feeds MC event shock
    binary_event: bool = False  # triggers jump-diffusion MC if True
    data_sufficiency: float = Field(ge=0.0, le=1.0)  # overconfidence detection
    data_gaps: List[str]


class MetaAgentReport(BaseModel):
    """Supervisory analysis of all agent outputs before CIO aggregation."""

    opportunity_id: str
    agent_weights: Dict[str, float]  # {agent_id: weight}
    direction_consensus: float  # fraction agreeing on direction
    score_spread: float  # std deviation of agent scores
    overconfidence_flag: bool
    anchoring_bias: Optional[str] = None  # description if detected
    dissent_bonus: Dict[str, float] = Field(default_factory=dict)  # {agent_id: bonus}
    risk_adjustment: float = 1.0  # position size multiplier (0-1)
    regime: str = "default"  # momentum | event | risk_off | default


class MonteCarloResult(BaseModel):
    """Output of 10,000-path Monte Carlo simulation."""

    expected_value: float  # mean terminal price
    median_value: float
    prob_profit: float  # P(return > 0)
    prob_loss_10: float  # P(return < -10%)
    prob_gain_20: float  # P(return > +20%)
    prob_gain_50: float  # P(return > +50%)
    var_95: float  # Value-at-Risk at 95% confidence (negative)
    var_99: float  # Value-at-Risk at 99% confidence (negative)
    max_drawdown_median: float  # median max drawdown across paths
    avg_profit_pct: float  # average profit when profitable
    avg_loss_pct: float  # average loss when loss-making
    percentiles: Dict[str, float]  # {p5, p25, p50, p75, p95}
    n_simulations: int
    time_horizon_days: int


class Scenario(BaseModel):
    """Named scenario for decision narrative."""

    name: str  # "Bull case" | "Bear case" | "Base case"
    probability: float
    target_price: float
    return_pct: float
    narrative: str
    key_driver: str


class CIODecisionV2(BaseModel):
    """CIO 2.0 probabilistic decision (V2).

    Produced by CIO v2 after Monte Carlo + Kelly + risk adjustment.
    """

    opportunity_id: str
    ticker: str
    decision: Literal["BUY", "SELL", "MONITOR", "PASS"]
    weighted_score: float
    position_size_pct: float = Field(ge=0.0, le=10.0)  # % of portfolio
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: float
    kelly_fraction: float
    monte_carlo: MonteCarloResult
    scenarios: List[Scenario]
    agent_weights_used: Dict[str, float]  # audit trail
    veto_triggered: bool = False
    veto_reason: Optional[str] = None
    risk_agent_score: Optional[int] = None
    meta_risk_adjustment: float = 1.0


# ===========================================================================
# Execution schemas
# ===========================================================================


class Order(BaseModel):
    """A paper trading order."""

    order_id: str
    opportunity_id: str
    ticker: str
    order_type: Literal["MARKET", "LIMIT"]
    side: Literal["BUY", "SELL"]
    shares: int
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: Literal["PENDING", "FILLED", "CANCELLED"] = "PENDING"
    created_at: str  # ISO datetime


class Position(BaseModel):
    """An open paper trading position."""

    position_id: str
    opportunity_id: str
    ticker: str
    entry_price: float
    shares: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    current_stop: Optional[float] = None
    entry_date: str  # ISO datetime
    decision_id: str


class ClosedTrade(BaseModel):
    """A completed paper trade."""

    trade_id: str
    opportunity_id: str
    ticker: str
    entry_price: float
    exit_price: float
    shares: int
    pnl_pct: float
    pnl_dollar: float
    hold_days: int
    exit_reason: Literal["STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", "MANUAL", "EXPIRED"]
    entry_date: str
    exit_date: str
    decision_id: str


# ===========================================================================
# Learning schemas
# ===========================================================================


class TradeEvaluation(BaseModel):
    """Post-trade evaluation for agent calibration."""

    trade_id: str
    opportunity_id: str
    ticker: str
    direction_correct: bool
    predicted_return_pct: float  # from MC expected_value
    actual_return_pct: float
    magnitude_error: float  # abs(predicted - actual)
    agent_scores: Dict[str, int]  # {agent_id: score} at decision time
    agent_directions: Dict[str, str]  # {agent_id: LONG|SHORT|FLAT}
    which_agents_correct: Dict[str, bool]  # {agent_id: correct?}


class AgentPerformanceStats(BaseModel):
    """Rolling performance stats per agent."""

    agent_id: str
    n_trades: int
    hit_rate: float  # direction accuracy
    avg_magnitude_error: float
    calibration_score: float  # how well confidence matches outcomes
    current_weight: float
    weight_history: List[float]  # last 20 weight adjustments
