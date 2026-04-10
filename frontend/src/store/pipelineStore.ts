import { create } from 'zustand'
import {
  type Node,
  type Edge,
  type NodeChange,
  applyNodeChanges,
} from '@xyflow/react'
import { getLayoutedElements } from '@/lib/dagre'
import type { FeedItem, Opportunity } from '@/types/pipeline'

// ─── Types ─────────────────────────────────────────────────────────────────

export type NodeStatus = 'idle' | 'running' | 'complete' | 'error'

export interface PipelineNodeData extends Record<string, unknown> {
  label: string
  status: NodeStatus
  lastResult?: string
  conviction?: number
}

// Re-export shared types for convenience
export type { FeedItem, Opportunity }

type PipelineNode = Node<PipelineNodeData>

// ─── Initial pipeline topology (V2) ────────────────────────────────────────

const rawNodes: PipelineNode[] = [
  { id: 'scanner',    type: 'stage', position: { x: 0, y: 0 }, data: { label: 'Market Scanner',    status: 'idle' } },
  { id: 'signal_detector', type: 'stage', position: { x: 0, y: 0 }, data: { label: 'Signal Detector', status: 'idle' } },
  { id: 'quality_gate', type: 'gate',  position: { x: 0, y: 0 }, data: { label: 'Quality Gate',    status: 'idle' } },
  // V2 quant agents
  { id: 'momentum',  type: 'agent', position: { x: 0, y: 0 }, data: { label: 'Momentum',  status: 'idle' } },
  { id: 'value',     type: 'agent', position: { x: 0, y: 0 }, data: { label: 'Value',     status: 'idle' } },
  { id: 'event',     type: 'agent', position: { x: 0, y: 0 }, data: { label: 'Event',     status: 'idle' } },
  { id: 'macro',     type: 'agent', position: { x: 0, y: 0 }, data: { label: 'Macro',     status: 'idle' } },
  { id: 'risk',      type: 'agent', position: { x: 0, y: 0 }, data: { label: 'Risk',      status: 'idle' } },
  { id: 'committee', type: 'stage', position: { x: 0, y: 0 }, data: { label: 'Meta-Agent', status: 'idle' } },
  { id: 'cio',       type: 'stage', position: { x: 0, y: 0 }, data: { label: 'CIO v2',   status: 'idle' } },
]

const rawEdges: Edge[] = [
  { id: 'e-scanner-signal',    type: 'animated', source: 'scanner',    target: 'signal_detector', data: { active: false } },
  { id: 'e-signal-gate',       type: 'animated', source: 'signal_detector', target: 'quality_gate', data: { active: false } },
  { id: 'e-gate-momentum',     type: 'animated', source: 'quality_gate', target: 'momentum',  data: { active: false } },
  { id: 'e-gate-value',        type: 'animated', source: 'quality_gate', target: 'value',     data: { active: false } },
  { id: 'e-gate-event',        type: 'animated', source: 'quality_gate', target: 'event',     data: { active: false } },
  { id: 'e-gate-macro',        type: 'animated', source: 'quality_gate', target: 'macro',     data: { active: false } },
  { id: 'e-gate-risk',         type: 'animated', source: 'quality_gate', target: 'risk',      data: { active: false } },
  { id: 'e-momentum-committee',type: 'animated', source: 'momentum',  target: 'committee', data: { active: false } },
  { id: 'e-value-committee',   type: 'animated', source: 'value',     target: 'committee', data: { active: false } },
  { id: 'e-event-committee',   type: 'animated', source: 'event',     target: 'committee', data: { active: false } },
  { id: 'e-macro-committee',   type: 'animated', source: 'macro',     target: 'committee', data: { active: false } },
  { id: 'e-risk-committee',    type: 'animated', source: 'risk',      target: 'committee', data: { active: false } },
  { id: 'e-committee-cio',     type: 'animated', source: 'committee', target: 'cio',       data: { active: false } },
]

const { nodes: initialNodes, edges: initialEdges } = getLayoutedElements(rawNodes, rawEdges, 'LR')

// ─── Helpers ────────────────────────────────────────────────────────────────

function sortAndCap(rankings: Opportunity[]): Opportunity[] {
  return [...rankings]
    .sort((a, b) => b.convictionScore - a.convictionScore)
    .slice(0, 10)
}

/** Derive risk rating from risk agent score (0-100, higher = safer) */
function riskRatingFromScore(score: number | undefined): string {
  if (score === undefined) return 'MEDIUM'
  if (score >= 70) return 'LOW'
  if (score >= 40) return 'MEDIUM'
  return 'HIGH'
}

// ─── Store ──────────────────────────────────────────────────────────────────

interface PipelineState {
  nodes: PipelineNode[]
  edges: Edge[]
  feedItems: FeedItem[]
  outputRankings: Opportunity[]
  selectedOpportunityId: string | null
}

interface PipelineActions {
  onNodesChange: (changes: NodeChange<PipelineNode>[]) => void
  updateNodeStatus: (nodeId: string, status: NodeStatus, data?: Partial<PipelineNodeData>) => void
  addFeedItem: (item: FeedItem) => void
  setOutputRankings: (rankings: Opportunity[]) => void
  handleSSEEvent: (eventType: string, data: Record<string, unknown>) => void
  setSelectedOpportunity: (id: string | null) => void
}

type PipelineStore = PipelineState & PipelineActions

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  nodes: initialNodes,
  edges: initialEdges,
  feedItems: [],
  outputRankings: [],
  selectedOpportunityId: null,

  onNodesChange: (changes) => {
    set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) }))
  },

  updateNodeStatus: (nodeId, status, extra) => {
    set((state) => {
      const isRunning = status === 'running'
      return {
        nodes: state.nodes.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, status, ...extra } } : n
        ),
        edges: state.edges.map((e) =>
          e.source === nodeId ? { ...e, data: { ...e.data, active: isRunning } } : e
        ),
      }
    })
  },

  addFeedItem: (item) => {
    set((state) => ({ feedItems: [item, ...state.feedItems].slice(0, 100) }))
  },

  setOutputRankings: (rankings) => {
    set({ outputRankings: sortAndCap(rankings) })
  },

  setSelectedOpportunity: (id) => {
    set({ selectedOpportunityId: id })
  },

  handleSSEEvent: (eventType, data) => {
    const { updateNodeStatus, addFeedItem } = get()

    switch (eventType) {
      case 'AGENT_STARTED': {
        // V2: { agent_id, opportunity_id, ticker }
        const agentId = (data['agent_id'] ?? data['persona']) as string | undefined
        if (agentId) updateNodeStatus(agentId, 'running')
        break
      }

      case 'AGENT_COMPLETE': {
        // V2: { agent_id, opportunity_id, ticker, score, direction, conviction }
        const agentId = (data['agent_id'] ?? data['persona']) as string | undefined
        const direction = (data['direction'] ?? data['verdict']) as string | undefined
        const score = data['score'] != null ? Number(data['score']) : undefined
        const ticker = data['ticker'] as string | undefined

        if (agentId) {
          updateNodeStatus(agentId, 'complete', {
            lastResult: direction ?? undefined,
          })
        }
        if (ticker) {
          addFeedItem({
            id: `${agentId ?? 'agent'}-${ticker}-${Date.now()}`,
            type: 'detection',
            ticker,
            headline: `${agentId ?? 'Agent'} completed analysis — ${direction ?? ''}`,
            convictionScore: score,
            timestamp: Date.now(),
          })
        }
        break
      }

      case 'COMMITTEE_COMPLETE': {
        updateNodeStatus('committee', 'complete')
        break
      }

      case 'RISK_VETO': {
        // V2: { opportunity_id, ticker, veto_reason }
        const ticker = (data['ticker'] as string) ?? 'UNKNOWN'
        addFeedItem({
          id: `veto-${ticker}-${Date.now()}`,
          type: 'rejection',
          ticker,
          headline: `Risk veto triggered`,
          rejectionReason: (data['veto_reason'] as string) ?? 'Risk score too low',
          timestamp: Date.now(),
        })
        break
      }

      case 'DECISION_MADE': {
        // V2 CIODecisionV2: { opportunity_id, ticker, decision: CIODecisionV2, verdicts: [...] }
        updateNodeStatus('cio', 'complete')
        const decisionBlob = data['decision'] as Record<string, unknown> | undefined
        const ticker = (data['ticker'] as string) ?? 'UNKNOWN'
        const opportunityId = (data['opportunity_id'] as string) ?? crypto.randomUUID()

        if (decisionBlob) {
          // CIODecisionV2 fields
          const finalVerdict = (decisionBlob['decision'] as string) ?? 'UNKNOWN'
          const weightedScore = Number(decisionBlob['weighted_score'] ?? 0)
          const positionSizePct = Number(decisionBlob['position_size_pct'] ?? 0)
          const riskAgentScore = decisionBlob['risk_agent_score'] != null
            ? Number(decisionBlob['risk_agent_score'])
            : undefined
          const riskRating = riskRatingFromScore(riskAgentScore)
          const vetoTriggered = Boolean(decisionBlob['veto_triggered'])

          const isApproval = new Set(['BUY', 'MONITOR']).has(finalVerdict) && !vetoTriggered

          addFeedItem({
            id: opportunityId,
            type: isApproval ? 'decision' : 'rejection',
            ticker,
            headline: isApproval
              ? `${finalVerdict} — score ${weightedScore.toFixed(0)}`
              : `Rejected: ${finalVerdict}`,
            convictionScore: isApproval ? Math.round(weightedScore) : undefined,
            riskRating,
            finalVerdict,
            rejectionReason: isApproval ? undefined : finalVerdict,
            timestamp: Date.now(),
          })

          // Build verdicts from V2 QuantAgentVerdict
          const rawVerdicts = data['verdicts'] as Array<Record<string, unknown>> | undefined
          const agentScores = (rawVerdicts ?? []).map((v) => ({
            persona: (v['agent_id'] as string) ?? (v['persona'] as string) ?? '',
            verdict: (v['direction'] as string) ?? (v['verdict'] as string) ?? '',
            confidence: Number(v['score'] ?? v['confidence'] ?? 0),
            rationale: v['bull_factors'] != null
              ? (v['bull_factors'] as string[]).join('; ')
              : (v['rationale'] as string | undefined),
          }))

          const opportunity: Opportunity = {
            opportunityId,
            ticker,
            convictionScore: Math.round(weightedScore),
            suggestedAllocationPct: positionSizePct,
            finalVerdict,
            riskRating,
            decidedAt: new Date().toISOString(),
            agentScores,
            timeHorizon: decisionBlob['monte_carlo'] != null
              ? `${(decisionBlob['monte_carlo'] as Record<string, unknown>)['time_horizon_days']}d`
              : undefined,
          }

          set((state) => {
            const filtered = state.outputRankings.filter(
              (o) => o.opportunityId !== opportunityId
            )
            return { outputRankings: sortAndCap([opportunity, ...filtered]) }
          })
        }
        break
      }

      default:
        break
    }
  },
}))
