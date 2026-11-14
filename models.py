"""Data contracts and state definitions for Payment Compliance Gatekeeper."""

from typing import List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class PaymentTransaction(BaseModel):
    tx_id: str
    originator_id: str
    beneficiary_id: str
    amount_usd: float
    currency: str = "USD"
    purpose_code: str
    cross_border: bool
    # Free text supplied by the payer (the remittance / payment-details line of the
    # payment message). The rule path never reads it; the Gemini path does.
    remittance_info: str = ""


class OwnershipPath(BaseModel):
    ubo_id: str
    ubo_name: str
    jurisdiction: str
    is_sanctioned: bool
    sanction_list: Optional[str] = None
    hop_count: int
    effective_ownership_pct: float = Field(
        description="Effective ownership percentage along the path (0.0 to 1.0)"
    )
    path_entities: List[str] = Field(
        description="Entity ID chain from beneficiary to ultimate beneficial owner (UBO)"
    )


# [1. Multi-source Observability] Unified telemetry observation model
class TelemetryObservation(BaseModel):
    obs_id: str
    source: Literal["SPANNER_GRAPH", "METRICS", "TRACES", "LOGS"]
    entity_or_tx_id: str
    timestamp: str
    summary: str
    is_anomalous: bool


# [3. Hypothesis Verification] Explicit compliance/fault hypothesis model
class ComplianceHypothesis(BaseModel):
    hypothesis_id: str
    description: str
    required_tool: Literal[
        "query_payment_metrics",
        "trace_beneficial_owners",
        "query_swift_traces",
        "query_kyc_audit_logs",
    ]
    status: Literal["PENDING", "CONFIRMED", "REFUTED"] = "PENDING"
    supporting_obs_ids: List[str] = Field(default_factory=list)


class RegulatoryRule(BaseModel):
    rule_id: str
    title: str
    content: str
    similarity_score: float


# [5. RAG + Grounding] Operational remediation runbook model
class ComplianceRunbook(BaseModel):
    runbook_id: str
    title: str
    trigger_condition: str
    action_steps: List[str]


class ComplianceDecision(BaseModel):
    decision: Literal["APPROVE", "BLOCK", "MANUAL_ESCALATION"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confirmed_hypothesis_ids: List[str] = Field(
        description="Hypothesis IDs verified as CONFIRMED via dynamic tool probing"
    )
    correlated_observation_ids: List[str] = Field(
        description="Correlated telemetry observation IDs across multiple sources"
    )
    cited_rule_ids: List[str] = Field(
        description="Strictly grounded rule_ids from retrieved RegulatoryRules"
    )
    cited_runbook_ids: List[str] = Field(
        description="Strictly grounded runbook_ids from retrieved ComplianceRunbooks"
    )
    flagged_entity_ids: List[str] = Field(
        description="Strictly grounded entity_ids from Spanner Graph ownership paths"
    )
    calculated_sanctioned_ownership_pct: float = Field(
        description="Aggregate effective ownership by sanctioned entities (0.0 to 1.0)"
    )
    remediation_actions: List[str] = Field(
        description="Concrete remediation actions grounded in the cited Runbook"
    )
    rationale: str


class GatekeeperState(TypedDict):
    transaction: PaymentTransaction
    hypotheses: List[ComplianceHypothesis]
    observations: List[TelemetryObservation]
    tool_call_trajectory: List[str]
    ownership_paths: List[OwnershipPath]
    retrieved_rules: List[RegulatoryRule]
    retrieved_runbooks: List[ComplianceRunbook]
    # IDs (rules and runbooks) that were retrieved only because of the remittance
    # text. The rule path ignores them, which keeps it text-blind.
    text_driven_ids: List[str]
    draft_decision: Optional[ComplianceDecision]
    provenance_valid: bool
    provenance_errors: List[str]
    # One entry per pass through the provenance gate: the list of errors found
    # (empty list = the draft passed). Lets the evaluation count first-pass rate.
    gate_history: List[List[str]]
    retry_count: int
    final_decision: Optional[ComplianceDecision]
