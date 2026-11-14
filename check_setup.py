"""Preflight: check that each cloud piece is reachable before running the comparison.

    python check_setup.py              packages, credentials, one Gemini call, one embedding call
    python check_setup.py --spanner    also check the Spanner database and the graph query

Each line ends in PASS, FAIL or SKIP. Paste the whole output back if anything fails.
The two model calls cost a fraction of a cent.
"""

import argparse
import importlib
import os
import sys
import time

from config import load_config

results = []


def report(name: str, status: str, detail: str = "") -> None:
    results.append(status)
    print(f"[{status:<4}] {name}" + (f" - {detail}" if detail else ""))


def check_packages() -> None:
    wanted = [
        ("google.genai", "google-genai", True),
        ("langgraph", "langgraph", False),
        ("faiss", "faiss-cpu", False),
        ("braintrust", "braintrust", False),
        ("google.cloud.spanner", "google-cloud-spanner", False),
        ("vertexai", "google-cloud-aiplatform[evaluation]", False),
    ]
    for module, package, required in wanted:
        try:
            importlib.import_module(module)
            report(f"package {package}", "PASS")
        except ImportError:
            fallback = {
                "langgraph": "the built-in state machine is used instead",
                "faiss-cpu": "the NumPy index is used instead",
            }.get(package, "needed only for that service")
            report(f"package {package}", "FAIL" if required else "SKIP", f"not installed; {fallback}")


def check_credentials(cfg) -> bool:
    if not cfg.project:
        report("GOOGLE_CLOUD_PROJECT", "FAIL", 'not set. PowerShell: $env:GOOGLE_CLOUD_PROJECT = "your-project-id"')
        return False
    report("GOOGLE_CLOUD_PROJECT", "PASS", cfg.project)
    try:
        import google.auth

        _creds, adc_project = google.auth.default()
        report("Application Default Credentials", "PASS", f"found (quota project: {adc_project})")
        return True
    except Exception as exc:  # noqa: BLE001
        report("Application Default Credentials", "FAIL",
               f"{type(exc).__name__}. Run: gcloud auth application-default login")
        return False


def check_gemini(cfg) -> None:
    name = f"Gemini decision call ({cfg.gemini_model} @ {cfg.gemini_location})"
    try:
        from gemini_client import make_genai_client
        from pydantic import BaseModel

        class Ping(BaseModel):
            status: str

        client = make_genai_client(cfg.project, cfg.gemini_location)
        config = {"response_mime_type": "application/json", "response_schema": Ping}
        if cfg.thinking_level:
            config["thinking_config"] = {"thinking_level": cfg.thinking_level}
        started = time.perf_counter()
        response = client.models.generate_content(
            model=cfg.gemini_model,
            contents='Return the JSON object {"status": "ok"}.',
            config=config,
        )
        elapsed = time.perf_counter() - started
        usage = response.usage_metadata
        parsed = response.parsed
        ok = isinstance(parsed, Ping)
        report(name, "PASS" if ok else "FAIL",
               f"structured output {'parsed' if ok else 'did NOT parse: ' + repr(response.text)[:120]}; "
               f"{usage.prompt_token_count} in / {usage.candidates_token_count} out / "
               f"{usage.thoughts_token_count or 0} thinking tokens; {elapsed:.1f}s")
    except Exception as exc:  # noqa: BLE001
        hint = ""
        code = getattr(exc, "code", None)
        if code == 404:
            hint = " | model not served in this location: try PCG_GEMINI_LOCATION=global (or us)"
        elif code == 403:
            hint = " | enable the API (aiplatform.googleapis.com) and check the project id"
        elif code == 429:
            hint = " | quota exhausted; trial accounts have low limits for generative models"
        report(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}{hint}")


def check_embeddings(cfg) -> None:
    name = f"Gemini embedding call ({cfg.embed_model} @ {cfg.embed_location}, dim {cfg.embed_dim})"
    try:
        from gemini_client import make_genai_client

        client = make_genai_client(cfg.project, cfg.embed_location)
        response = client.models.embed_content(
            model=cfg.embed_model,
            contents="Payment to a consignee in a sanctioned territory",
            config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": cfg.embed_dim},
        )
        got = len(response.embeddings[0].values)
        report(name, "PASS" if got == cfg.embed_dim else "FAIL", f"returned {got} dimensions")
    except Exception as exc:  # noqa: BLE001
        report(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}")


def check_retrieval(cfg) -> None:
    """End to end: embed the corpus, then see what a sanctions-nexus sentence retrieves."""
    name = "retrieval with real embeddings"
    try:
        from faiss_regulatory_store import FAISSRegulatoryRetriever
        from gemini_client import GeminiEmbedder

        retriever = FAISSRegulatoryRetriever(embedder=GeminiEmbedder(cfg))
        hits = retriever.search_rules("goods for onward shipment to a consignee in Crimea", top_k=2)
        top = ", ".join(f"{h.rule_id} ({h.similarity_score})" for h in hits)
        ok = hits and hits[0].rule_id == "REG-SANCT-JURIS"
        report(name, "PASS" if ok else "FAIL",
               f"top rules for a Crimea shipment sentence: {top}"
               + ("" if ok else " | expected REG-SANCT-JURIS first"))
    except Exception as exc:  # noqa: BLE001
        report(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}")


def check_spanner(cfg) -> None:
    name = f"Spanner graph query ({cfg.spanner_instance}/{cfg.spanner_database})"
    try:
        from spanner_graph_client import SpannerGraphComplianceClient

        cloud = SpannerGraphComplianceClient(
            cfg.project, cfg.spanner_instance, cfg.spanner_database, use_mock=False
        )
        local = SpannerGraphComplianceClient(use_mock=True)
        cloud_paths, _ = cloud.trace_beneficial_owners("ENT-BEN-9001")
        local_paths, _ = local.trace_beneficial_owners("ENT-BEN-9001")
        pct = cloud.aggregate_sanctioned_ownership(cloud_paths)
        same = [p.model_dump() for p in cloud_paths] == [p.model_dump() for p in local_paths]
        if cloud.query_mode == "sql_fallback":
            report(name, "FAIL",
                   f"GQL failed, plain SQL was used instead. GQL error: {cloud.last_query_error}")
        else:
            report(name, "PASS" if same else "FAIL",
                   f"ENT-BEN-9001: {len(cloud_paths)} paths, {pct:.1%} sanctioned, "
                   f"{'matches' if same else 'DIFFERS from'} the local graph")
    except Exception as exc:  # noqa: BLE001
        report(name, "FAIL",
               f"{type(exc).__name__}: {str(exc)[:300]} | has seed_spanner.py been run?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spanner", action="store_true")
    args = parser.parse_args()
    cfg = load_config()

    check_packages()
    if check_credentials(cfg):
        check_gemini(cfg)
        check_embeddings(cfg)
        check_retrieval(cfg)
        if args.spanner:
            check_spanner(cfg)
    report("BRAINTRUST_API_KEY", "PASS" if os.environ.get("BRAINTRUST_API_KEY") else "SKIP",
           "set" if os.environ.get("BRAINTRUST_API_KEY")
           else "not set; Braintrust runs locally and sends nothing")

    failed = results.count("FAIL")
    print(f"\n{results.count('PASS')} passed, {failed} failed, {results.count('SKIP')} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
