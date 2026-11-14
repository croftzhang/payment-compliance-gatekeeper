"""Comprehensive unit and integration test suite for Payment Compliance Gatekeeper."""

import unittest
from models import (
    ComplianceDecision,
    PaymentTransaction,
)
from spanner_graph_client import SpannerGraphComplianceClient
from faiss_regulatory_store import FAISSRegulatoryRetriever
from compliance_graph_agent import PaymentComplianceGatekeeper
from braintrust_eval import (
    EVAL_DATASET,
    run_evaluation_suite,
    rca_and_decision_accuracy_scorer,
    dynamic_tool_trajectory_scorer,
    evidence_grounding_and_reliability_scorer,
)


class TestPaymentComplianceGatekeeper(unittest.TestCase):
    def setUp(self):
        self.gatekeeper = PaymentComplianceGatekeeper(use_vertex_llm=False)

    def test_1_multi_source_observability_probes(self):
        """Verify Spanner Graph, Metrics, Traces, and Logs return structured TelemetryObservation."""
        client = SpannerGraphComplianceClient(use_mock=True)
        tx = PaymentTransaction(**EVAL_DATASET[0]["input"])

        paths, graph_obs = client.trace_beneficial_owners(tx.beneficiary_id)
        met_obs = client.query_payment_metrics(tx)
        trc_obs = client.query_swift_traces(tx)
        log_obs = client.query_kyc_audit_logs(tx.beneficiary_id)

        self.assertEqual(len(paths), 3)
        self.assertAlmostEqual(client.aggregate_sanctioned_ownership(paths), 0.52, places=4)
        self.assertEqual(graph_obs[0].source, "SPANNER_GRAPH")
        self.assertEqual(met_obs[0].source, "METRICS")
        self.assertEqual(trc_obs[0].source, "TRACES")
        self.assertEqual(log_obs[0].source, "LOGS")
        self.assertTrue(all(o.is_anomalous for o in [graph_obs[0], met_obs[0], trc_obs[0], log_obs[0]]))

    def test_2_dynamic_tool_calling_and_hypothesis_spawning(self):
        """Verify high-risk cross-border tx dynamically spawns HYP-KYC-STRUCTURING while low-risk domestic skips extra tools."""
        # Case A: Sanctioned multi-hop shell UBO (TX-2026-9981) -> 4 tools including dynamically spawned KYC log check
        high_risk_tx = PaymentTransaction(**EVAL_DATASET[0]["input"])
        state_high = self.gatekeeper.run(high_risk_tx)
        self.assertEqual(
            state_high["tool_call_trajectory"],
            [
                "query_payment_metrics",
                "trace_beneficial_owners",
                "query_swift_traces",
                "query_kyc_audit_logs",
            ],
        )
        self.assertEqual(state_high["final_decision"].decision, "BLOCK")

        # Case B: Low-risk domestic wire (TX-2026-1004) -> Only 2 tools called, skips SWIFT traces and KYC logs
        low_risk_tx = PaymentTransaction(**EVAL_DATASET[2]["input"])
        state_low = self.gatekeeper.run(low_risk_tx)
        self.assertEqual(
            state_low["tool_call_trajectory"],
            ["query_payment_metrics", "trace_beneficial_owners"],
        )
        self.assertEqual(state_low["final_decision"].decision, "APPROVE")

    def test_3_moderate_risk_cross_border_escalation(self):
        """Verify cross-border wire with anomalous SWIFT trace but 0% sanctioned UBO routes to MANUAL_ESCALATION without KYC structuring spawn."""
        mod_tx = PaymentTransaction(**EVAL_DATASET[1]["input"])
        state_mod = self.gatekeeper.run(mod_tx)
        self.assertEqual(
            state_mod["tool_call_trajectory"],
            [
                "query_payment_metrics",
                "trace_beneficial_owners",
                "query_swift_traces",
            ],
        )
        self.assertEqual(state_mod["final_decision"].decision, "MANUAL_ESCALATION")
        self.assertIn("RB-EDD-ESCALATE-02", state_mod["final_decision"].cited_runbook_ids)

    def test_4_rag_dual_retrieval_rules_and_runbooks(self):
        """Verify FAISS retriever returns matching RegulatoryRule and ComplianceRunbook."""
        retriever = FAISSRegulatoryRetriever()
        rules = retriever.search_rules("50 percent aggregate blocked shell entities OFAC", top_k=2)
        rbs = retriever.search_runbooks("50 percent aggregate blocked shell entities OFAC", top_k=1)

        self.assertTrue(any(r.rule_id == "REG-OFAC-50PCT" for r in rules))
        self.assertEqual(rbs[0].runbook_id, "RB-OFAC-FREEZE-01")

    def test_5_deterministic_grounding_gate_catches_hallucinations(self):
        """Verify validate_provenance catches hallucinated rule_id, runbook_id, single-source evidence, and ownership math error."""
        tx = PaymentTransaction(**EVAL_DATASET[0]["input"])
        state = self.gatekeeper.run(tx)

        # Inject a tampered draft decision with hallucinated IDs, single observation source, and wrong math
        bad_decision = ComplianceDecision(
            decision="APPROVE",  # Policy violation when sanctioned >= 50%
            risk_level="LOW",
            confirmed_hypothesis_ids=["HYP-FAKE"],
            correlated_observation_ids=[state["observations"][0].obs_id],  # Only 1 source!
            cited_rule_ids=["REG-HALLUCINATED-999"],
            cited_runbook_ids=["RB-HALLUCINATED-999"],
            flagged_entity_ids=["ENT-NONEXISTENT-999"],
            calculated_sanctioned_ownership_pct=0.15,  # Math mismatch (actual is 0.52)
            remediation_actions=["None"],
            rationale="Tampered decision for unit test.",
        )
        state["draft_decision"] = bad_decision
        validation_res = self.gatekeeper.validate_provenance(state)

        self.assertFalse(validation_res["provenance_valid"])
        errors_joined = " | ".join(validation_res["provenance_errors"])
        self.assertIn("Ungrounded rule_id", errors_joined)
        self.assertIn("Ungrounded runbook_id", errors_joined)
        self.assertIn("at least 2 distinct telemetry sources", errors_joined)
        self.assertIn("Hypothesis mismatch", errors_joined)
        self.assertIn("Ungrounded entity_id", errors_joined)
        self.assertIn("Ownership math mismatch", errors_joined)
        self.assertIn("Policy violation", errors_joined)

    def test_6_automated_evaluation_suite_all_pass(self):
        """Verify all benchmark cases achieve 1.0 across RCA Accuracy, Tool Trajectory, and Evidence Grounding."""
        df = run_evaluation_suite(run_cloud_vertex_eval=False)
        self.assertEqual(len(df), 3)
        self.assertTrue((df["rca_and_decision_accuracy"] == 1.0).all())
        self.assertTrue((df["dynamic_tool_trajectory"] == 1.0).all())
        self.assertTrue((df["evidence_grounding_reliability"] == 1.0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
