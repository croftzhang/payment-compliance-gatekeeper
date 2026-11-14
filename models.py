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
    # The beneficiary as written on the payment message. When the name is empty the
    # screening step uses the name and country on the beneficiary's own record.
    beneficiary_name: str = ""
    beneficiary_country: str = ""
    beneficiary_registration_no: str = ""


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


class NameScreeningHit(BaseModel):
    """A sanctions-list entry whose name resembles a party on the payment."""

    list_id: str
    listed_name: str
    matched_on: str = Field(description="The listed name or alias that scored highest")
    party_role: str
    party_name: str
    score: float = Field(description="Name similarity, 0.0 to 1.0")
    level: Literal["ALERT", "NEAR_MISS"] = Field(
        description="ALERT: at or above the screening threshold. NEAR_MISS: just below it; "
        "code takes no action on these"
    )
    listed_country: Optional[str] = None
    listed_registration_no: Optional[str] = None
    listed_program: Optional[str] = None
    country_check: Literal["match", "mismatch", "unknown", "not_checked"] = "not_checked"
    registration_check: Literal["match", "mismatch", "unknown", "not_checked"] = "not_checked"
    verdict: Literal["TRUE_MATCH", "FALSE_POSITIVE", "UNRESOLVED", "NOT_ADJUDICATED"] = "NOT_ADJUDICATED"
    verdict_reason: str = ""


# [1. Multi-source Observability] Unified telemetry observation model
class TelemetryObservation(BaseModel):
    obs_id: str
    source: Literal["SPANNER_GRAPH", "METRICS", "TRACES", "LOGS", "WATCHLIST"]
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
        "resolve_list_hits",
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
    matched_list_ids: List[str] = Field(
        default_factory=list,
        description="list_id of each sanctions-list entry judged to be the same party as "
        "a party on the payment; empty if none",
    )


class GatekeeperState(TypedDict):
    transaction: PaymentTransaction
    hypotheses: List[ComplianceHypothesis]
    observations: List[TelemetryObservation]
    tool_call_trajectory: List[str]
    ownership_paths: List[OwnershipPath]
    # Sanctions-list screening results for the beneficiary name: alerts and near misses.
    name_hits: List[NameScreeningHit]
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
