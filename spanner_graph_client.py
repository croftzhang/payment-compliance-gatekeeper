"""Spanner Graph ISO GQL client and multi-source observability probes.

Two backends answer `trace_beneficial_owners`:

  * use_mock=True  - an in-memory graph built from `reference_data.py`, traversed in
                     Python with the same semantics as the GQL query below.
  * use_mock=False - the ComplianceGraph property graph in Spanner, queried with GQL.

Both produce rows of the same shape and share the same post-processing, so their
results can be compared path for path (`python seed_spanner.py --verify-only`).

The Metrics, Traces and Logs probes are fixed responses keyed by beneficiary, as in
the original design. Only the ownership trace is backed by a real data store.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from models import OwnershipPath, PaymentTransaction, TelemetryObservation
import reference_data

try:
    from google.cloud import spanner
    SPANNER_AVAILABLE = True
except ImportError:
    SPANNER_AVAILABLE = False

MAX_HOPS = 4

# One row of the ownership query, whichever backend produced it:
# (ubo_id, ubo_name, jurisdiction, is_sanctioned, sanction_list, hop_count,
#  path_entities [beneficiary ... owner], edge_weights [nearest edge first])
RawPathRow = Tuple[str, str, str, bool, Optional[str], int, List[str], List[float]]


def enumerate_owner_paths(
    entities: Dict[str, dict],
    edges: Sequence[Tuple[str, str, float]],
    beneficiary_id: str,
    max_hops: int = MAX_HOPS,
) -> List[RawPathRow]:
    """Walk ownership edges upward from the beneficiary, mirroring the GQL query.

    Returns every acyclic path of 1..max_hops edges, including the ones that stop
    at an intermediate holder - exactly what `<-[:Owns]-{1,4}` with ACYCLIC returns.
    `edges` are (owner_id, target_id, ownership_pct).
    """
    owners_of: Dict[str, List[Tuple[str, float]]] = {}
    for owner_id, target_id, pct in edges:
        owners_of.setdefault(target_id, []).append((owner_id, pct))

    rows: List[RawPathRow] = []

    def walk(node_id: str, path: List[str], weights: List[float]) -> None:
        if len(weights) >= max_hops:
            return
        for owner_id, pct in sorted(owners_of.get(node_id, [])):
            if owner_id in path:          # ACYCLIC: never revisit a node
                continue
            new_path = path + [owner_id]
            new_weights = weights + [pct]
            ent = entities[owner_id]
            rows.append((
                owner_id, ent["entity_name"], ent["jurisdiction"], bool(ent["is_sanctioned"]),
                ent["sanction_list"], len(new_weights), new_path, new_weights,
            ))
            walk(owner_id, new_path, new_weights)

    walk(beneficiary_id, [beneficiary_id], [])
    return rows


class SpannerGraphComplianceClient:
    """Integrates Spanner Graph ISO GQL traversal with Metrics, Traces, and Logs probes."""

    # Spanner Graph (GoogleSQL dialect). Notes on what differs from Cypher-style GQL:
    #   * ACYCLIC is a path mode: paths that revisit a node are not returned, so
    #     cross-holdings between shells cannot loop.
    #   * NODES(p) / EDGES(p) return arrays; ARRAY_TRANSFORM projects a property.
    #     (There is no `[n IN nodes(p) | n.prop]` list comprehension.)
    #   * A quantified pattern returns every path of 1..4 hops, so intermediate
    #     holders come back too. Keeping only the outermost owner of each chain is
    #     done in `_rows_to_paths`, the same code the local backend goes through.
    ISO_GQL_UBO_QUERY = """
    GRAPH ComplianceGraph
    MATCH p = ACYCLIC (target:Entity)<-[:Owns]-{1,4}(owner:Entity)
    WHERE target.entity_id = @beneficiary_id
    RETURN
        owner.entity_id AS ubo_id,
        owner.entity_name AS ubo_name,
        owner.jurisdiction AS jurisdiction,
        owner.is_sanctioned AS is_sanctioned,
        owner.sanction_list AS sanction_list,
        PATH_LENGTH(p) AS hop_count,
        ARRAY_TRANSFORM(NODES(p), n -> n.entity_id) AS path_entities,
        ARRAY_TRANSFORM(EDGES(p), x -> x.ownership_pct) AS edge_weights
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        database_id: Optional[str] = None,
        use_mock: bool = True,
    ):
        self.use_mock = use_mock
        # "local" | "gql" | "sql_fallback" - how the last ownership trace was answered
        self.query_mode = "local"
        self.last_query_error: Optional[str] = None
        self.database = None
        if not self.use_mock:
            if not SPANNER_AVAILABLE:
                raise RuntimeError(
                    "The Spanner backend was requested but google-cloud-spanner is not "
                    "installed. Run: pip install -r requirements-cloud.txt"
                )
            if not (project_id and instance_id and database_id):
                raise RuntimeError(
                    "The Spanner backend needs project_id, instance_id and database_id."
                )
            client = spanner.Client(project=project_id)
            instance = client.instance(instance_id)
            self.database = instance.database(database_id)

    @staticmethod
    def _compute_chain_ownership(edge_weights: List[float]) -> float:
        product = 1.0
        for w in edge_weights:
            product *= w
        return round(product, 6)

    # ---------------- Tool 1: Spanner Graph ISO GQL UBO Trace ----------------
    def trace_beneficial_owners(
        self, beneficiary_id: str
    ) -> Tuple[List[OwnershipPath], List[TelemetryObservation]]:
        paths = (
            self._mock_iso_gql_execution(beneficiary_id)
            if self.use_mock
            else self._execute_gcp_spanner(beneficiary_id)
        )
        sanctioned_pct = self.aggregate_sanctioned_ownership(paths)
        is_anom = sanctioned_pct > 0.0
        obs = TelemetryObservation(
            obs_id=f"OBS-GRF-{beneficiary_id}",
            source="SPANNER_GRAPH",
            entity_or_tx_id=beneficiary_id,
            timestamp="2026-11-13T18:00:02Z",
            summary=(
                f"Spanner ISO GQL traced {len(paths)} UBO paths; "
                f"aggregate sanctioned ownership={sanctioned_pct:.1%}."
            ),
            is_anomalous=is_anom,
        )
        return paths, [obs]

    def _rows_to_paths(self, rows: Sequence[RawPathRow]) -> List[OwnershipPath]:
        """Keep the outermost owner of each chain and compute effective ownership.

        A row is dropped when another row extends it (same nodes, then more): that
        means its last node is an intermediate holder, not the end of the chain.
        """
        node_lists = [tuple(r[6]) for r in rows]
        paths: List[OwnershipPath] = []
        for row, nodes in zip(rows, node_lists):
            extended = any(
                len(other) > len(nodes) and other[: len(nodes)] == nodes
                for other in node_lists
            )
            if extended:
                continue
            ubo_id, ubo_name, jurisdiction, is_sanctioned, sanction_list, hop_count, _n, weights = row
            paths.append(
                OwnershipPath(
                    ubo_id=ubo_id,
                    ubo_name=ubo_name,
                    jurisdiction=jurisdiction,
                    is_sanctioned=bool(is_sanctioned),
                    sanction_list=sanction_list,
                    hop_count=int(hop_count),
                    effective_ownership_pct=self._compute_chain_ownership(list(weights)),
                    path_entities=list(nodes),
                )
            )
        paths.sort(key=lambda p: p.path_entities)
        return paths

    def _execute_gcp_spanner(self, beneficiary_id: str) -> List[OwnershipPath]:
        try:
            rows = self._run_gql(beneficiary_id)
            self.query_mode = "gql"
            self.last_query_error = None
        except Exception as exc:  # noqa: BLE001 - any query failure takes the same route
            # The graph query failed (most likely a syntax or feature problem). Read
            # the two base tables with plain SQL and traverse them here instead, and
            # keep the error so the caller can report it - this is never silent.
            self.query_mode = "sql_fallback"
            self.last_query_error = f"{type(exc).__name__}: {exc}"
            rows = self._run_sql_fallback(beneficiary_id)
        return self._rows_to_paths(rows)

    def _run_gql(self, beneficiary_id: str) -> List[RawPathRow]:
        rows: List[RawPathRow] = []
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                self.ISO_GQL_UBO_QUERY,
                params={"beneficiary_id": beneficiary_id},
                param_types={"beneficiary_id": spanner.param_types.STRING},
            )
            for row in results:
                (
                    ubo_id, ubo_name, jurisdiction, is_sanctioned,
                    sanction_list, hop_count, path_entities, edge_weights,
                ) = row
                rows.append((
                    ubo_id, ubo_name, jurisdiction, is_sanctioned, sanction_list,
                    int(hop_count), list(path_entities), [float(w) for w in edge_weights],
                ))
        return rows

    def _run_sql_fallback(self, beneficiary_id: str) -> List[RawPathRow]:
        with self.database.snapshot(multi_use=True) as snapshot:
            entities = {
                r[0]: {
                    "entity_name": r[1], "jurisdiction": r[2],
                    "is_sanctioned": r[3], "sanction_list": r[4],
                }
                for r in snapshot.execute_sql(
                    "SELECT entity_id, entity_name, jurisdiction, is_sanctioned, sanction_list "
                    "FROM Entities"
                )
            }
            edges = [
                (r[0], r[1], float(r[2]))
                for r in snapshot.execute_sql(
                    "SELECT owner_id, target_id, ownership_pct FROM EntityOwnership"
                )
            ]
        return enumerate_owner_paths(entities, edges, beneficiary_id)

    def _mock_iso_gql_execution(self, beneficiary_id: str) -> List[OwnershipPath]:
        """Local backend: the reference graph, traversed with the query's semantics."""
        self.query_mode = "local"
        entities = reference_data.entity_index()
        edges = [(o, t, pct) for o, t, pct, _eff in reference_data.OWNERSHIP]
        if beneficiary_id not in entities:
            return []
        return self._rows_to_paths(enumerate_owner_paths(entities, edges, beneficiary_id))

    # ---------------- Tool 2: Metrics Probe (Velocity & Structuring) ----------------
    def query_payment_metrics(self, tx: PaymentTransaction) -> List[TelemetryObservation]:
        if tx.beneficiary_id == "ENT-BEN-9001":
            return [
                TelemetryObservation(
                    obs_id=f"OBS-MET-{tx.tx_id}",
                    source="METRICS",
                    entity_or_tx_id=tx.beneficiary_id,
                    timestamp="2026-11-13T18:00:01Z",
                    summary="24h outbound wire volume spiked 640% above 90d baseline; velocity z-score=4.8.",
                    is_anomalous=True,
                )
            ]
        return [
            TelemetryObservation(
                obs_id=f"OBS-MET-{tx.tx_id}",
                source="METRICS",
                entity_or_tx_id=tx.beneficiary_id,
                timestamp="2026-11-13T18:00:01Z",
                summary="24h wire velocity within normal historical baseline (z-score=0.3).",
                is_anomalous=False,
            )
        ]

    # ---------------- Tool 3: Traces Probe (SWIFT Correspondent Routing) ----------------
    def query_swift_traces(self, tx: PaymentTransaction) -> List[TelemetryObservation]:
        if tx.beneficiary_id in ("ENT-BEN-9001", "ENT-BEN-7700"):
            return [
                TelemetryObservation(
                    obs_id=f"OBS-TRC-{tx.tx_id}",
                    source="TRACES",
                    entity_or_tx_id=tx.tx_id,
                    timestamp="2026-11-13T18:00:03Z",
                    summary=(
                        "SWIFT MT103 trace shows nested correspondent hop via high-risk "
                        "offshore intermediary bank (BIC: OFFSHOREVGXX) with stripped field 50F."
                    ),
                    is_anomalous=True,
                )
            ]
        return [
            TelemetryObservation(
                obs_id=f"OBS-TRC-{tx.tx_id}",
                source="TRACES",
                entity_or_tx_id=tx.tx_id,
                timestamp="2026-11-13T18:00:03Z",
                summary="Direct clearing span with complete originator metadata.",
                is_anomalous=False,
            )
        ]

    # ---------------- Tool 4: Logs Probe (KYC & Corporate Registry Audit) ----------------
    def query_kyc_audit_logs(self, beneficiary_id: str) -> List[TelemetryObservation]:
        if beneficiary_id == "ENT-BEN-9001":
            return [
                TelemetryObservation(
                    obs_id=f"OBS-LOG-{beneficiary_id}",
                    source="LOGS",
                    entity_or_tx_id=beneficiary_id,
                    timestamp="2026-11-13T17:42:10Z",
                    summary=(
                        "KYC Registry Audit Log: Equity split restructured 18 minutes prior "
                        "to wire initiation into two layers (ENT-SHELL-A 32%, ENT-SHELL-B 20%) "
                        "to mask single-path 50% threshold."
                    ),
                    is_anomalous=True,
                )
            ]
        return [
            TelemetryObservation(
                obs_id=f"OBS-LOG-{beneficiary_id}",
                source="LOGS",
                entity_or_tx_id=beneficiary_id,
                timestamp="2026-11-13T18:00:04Z",
                summary="No recent KYC beneficial ownership mutations in the past 365 days.",
                is_anomalous=False,
            )
        ]

    @staticmethod
    def aggregate_sanctioned_ownership(paths: List[OwnershipPath]) -> float:
        total = sum(p.effective_ownership_pct for p in paths if p.is_sanctioned)
        return round(min(total, 1.0), 4)
