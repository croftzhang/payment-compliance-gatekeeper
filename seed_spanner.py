"""Create the Spanner database if needed, load the ownership graph, and verify it.

Prerequisites: a Spanner instance (the free trial instance is enough) and
Application Default Credentials. See README.md, "Round 2".

    python seed_spanner.py                 create schema if missing, load data, verify
    python seed_spanner.py --verify-only   compare Spanner with the local graph, change nothing
    python seed_spanner.py --reset         drop the database first, then rebuild it

The data comes from reference_data.py, the same rows the local backend uses, so the
verification is a row-for-row comparison of the two backends for every beneficiary.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import reference_data
from config import load_config
from spanner_graph_client import SpannerGraphComplianceClient

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"
DDL_TIMEOUT_S = 600


def load_ddl_statements(path: Path = SCHEMA_FILE) -> List[str]:
    """schema.sql -> one string per statement, comments removed."""
    text = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
    return [" ".join(stmt.split()) for stmt in text.split(";") if stmt.strip()]


def _table_name(create_table_stmt: str) -> str:
    return create_table_stmt.split()[2]


def ensure_schema(instance, database_id: str, reset: bool):
    statements = load_ddl_statements()
    table_ddl = [s for s in statements if s.upper().startswith("CREATE TABLE")]
    graph_ddl = [s for s in statements if "PROPERTY GRAPH" in s.upper()]

    database = instance.database(database_id)
    if reset and database.exists():
        print(f"Dropping database '{database_id}' ...")
        database.drop()

    if not database.exists():
        print(f"Creating database '{database_id}' with {len(table_ddl)} tables and the property graph ...")
        database = instance.database(database_id, ddl_statements=table_ddl + graph_ddl)
        database.create().result(DDL_TIMEOUT_S)
        return database

    with database.snapshot() as snapshot:
        existing = {
            row[0]
            for row in snapshot.execute_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = ''"
            )
        }
    missing = [s for s in table_ddl if _table_name(s) not in existing]
    # CREATE OR REPLACE PROPERTY GRAPH is safe to run again.
    print(
        f"Database '{database_id}' exists; creating {len(missing)} missing table(s) "
        "and (re)defining the property graph ..."
    )
    database.update_ddl(missing + graph_ddl).result(DDL_TIMEOUT_S)
    return database


def load_data(database) -> None:
    from google.cloud import spanner

    with database.batch() as batch:
        # Edges first on delete, entities first on insert (foreign keys).
        batch.delete("EntityOwnership", spanner.KeySet(all_=True))
        batch.delete("Entities", spanner.KeySet(all_=True))
        batch.insert("Entities", reference_data.ENTITY_COLUMNS,
                     [list(row) for row in reference_data.ENTITIES])
        batch.insert("EntityOwnership", reference_data.OWNERSHIP_COLUMNS,
                     [list(row) for row in reference_data.OWNERSHIP])
    print(f"Loaded {len(reference_data.ENTITIES)} entities and "
          f"{len(reference_data.OWNERSHIP)} ownership edges.")


def _signature(paths):
    return sorted(
        (tuple(p.path_entities), round(p.effective_ownership_pct, 6), p.is_sanctioned, p.hop_count)
        for p in paths
    )


def verify(project: str, instance_id: str, database_id: str) -> int:
    """Returns 0 when Spanner (via GQL) matches the local graph for every beneficiary."""
    cloud = SpannerGraphComplianceClient(project, instance_id, database_id, use_mock=False)
    local = SpannerGraphComplianceClient(use_mock=True)
    all_match, used_fallback, gql_error = True, False, None

    print("\nbeneficiary     paths  sanctioned  answered_by   vs local graph")
    for beneficiary_id in reference_data.BENEFICIARY_IDS:
        cloud_paths, _ = cloud.trace_beneficial_owners(beneficiary_id)
        local_paths, _ = local.trace_beneficial_owners(beneficiary_id)
        match = _signature(cloud_paths) == _signature(local_paths)
        all_match &= match
        if cloud.query_mode == "sql_fallback":
            used_fallback, gql_error = True, cloud.last_query_error
        pct = cloud.aggregate_sanctioned_ownership(cloud_paths)
        print(f"{beneficiary_id:<15} {len(cloud_paths):>5}  {pct:>10.1%}  "
              f"{cloud.query_mode:<12}  {'MATCH' if match else 'DIFFERENT'}")

    if used_fallback:
        print("\nWARNING: the graph (GQL) query failed, so the paths above were read from the")
        print("base tables with plain SQL instead. The data is in Spanner, but Spanner Graph")
        print("itself was not exercised. The error was:\n")
        print(f"  {gql_error}\n")
        print("Send this error back and the query can be corrected.")
        return 2
    if not all_match:
        print("\nFAIL: Spanner and the local graph disagree.")
        return 1
    print("\nPASS: Spanner Graph (GQL) returns the same ownership paths as the local graph.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    project = cfg.require_project()
    try:
        from google.cloud import spanner
    except ImportError:
        print("google-cloud-spanner is not installed. Run: pip install -r requirements-cloud.txt")
        return 1

    print(f"Project: {project} | instance: {cfg.spanner_instance} | database: {cfg.spanner_database}")
    instance = spanner.Client(project=project).instance(cfg.spanner_instance)
    if not instance.exists():
        print(f"\nSpanner instance '{cfg.spanner_instance}' was not found in project '{project}'.")
        print("Create the free trial instance first (one per project, 90 days):\n")
        print(f"  gcloud spanner instances create {cfg.spanner_instance} "
              "--instance-type=free-instance --config=regional-us-central1 "
              f'--description="PCG trial" --project {project}')
        print("\nIf yours has another name, set PCG_SPANNER_INSTANCE to it.")
        return 1

    if not args.verify_only:
        database = ensure_schema(instance, cfg.spanner_database, args.reset)
        load_data(database)
    return verify(project, cfg.spanner_instance, cfg.spanner_database)


if __name__ == "__main__":
    sys.exit(main())
