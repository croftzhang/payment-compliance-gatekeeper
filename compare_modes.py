"""Run the same cases through the rule path and the Gemini path and compare them.

    python compare_modes.py                                   rule path only, fully offline
    python compare_modes.py --modes rules gemini --embeddings gemini --repeats 3
    python compare_modes.py --modes rules gemini --embeddings gemini --graph spanner \
                            --braintrust --vertex-eval

Both paths share the tools, the retrieval and the provenance gate. The only thing
that differs is who drafts the decision, so any difference in outcome is down to that.
The rule path is deterministic and runs once; the Gemini path runs --repeats times.

Results are printed and written to results/comparison_<timestamp>/.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import braintrust_eval as ev
from compliance_graph_agent import PaymentComplianceGatekeeper
from config import load_config
from faiss_regulatory_store import FAISSRegulatoryRetriever
from spanner_graph_client import SpannerGraphComplianceClient

STRATA_ORDER = ["core", "structured", "text_signal", "text_distractor"]

# First-attempt gate errors, grouped by what went wrong.
_ERROR_CATEGORIES = [
    ("Missing draft", "no usable draft"),
    ("rule_id", "rule citation"),
    ("runbook_id", "runbook citation"),
    ("observation_id", "evidence ids"),
    ("telemetry sources", "evidence from <2 sources"),
    ("Hypothesis mismatch", "hypothesis list"),
    ("entity_id", "entity ids"),
    ("Ownership math", "ownership arithmetic"),
    ("Policy violation", ">=50% not blocked"),
]


def categorise_error(message: str) -> str:
    for needle, label in _ERROR_CATEGORIES:
        if needle in message:
            return label
    return "other"


def build_gatekeepers(
    modes: List[str], graph: str, embeddings: str
) -> Dict[str, PaymentComplianceGatekeeper]:
    """One gatekeeper per mode, sharing the same graph client and retriever."""
    cfg = load_config()
    if graph == "spanner":
        spanner_client = SpannerGraphComplianceClient(
            cfg.require_project(), cfg.spanner_instance, cfg.spanner_database, use_mock=False
        )
    else:
        spanner_client = SpannerGraphComplianceClient(use_mock=True)

    if embeddings == "gemini":
        from gemini_client import GeminiEmbedder

        retriever = FAISSRegulatoryRetriever(embedder=GeminiEmbedder(cfg))
    else:
        retriever = FAISSRegulatoryRetriever()

    gatekeepers: Dict[str, PaymentComplianceGatekeeper] = {}
    for mode in modes:
        if mode == "rules":
            gatekeepers[mode] = PaymentComplianceGatekeeper(
                use_vertex_llm=False, spanner_client=spanner_client, retriever=retriever
            )
        elif mode == "gemini":
            from gemini_client import GeminiDecisionLLM

            gatekeepers[mode] = PaymentComplianceGatekeeper(
                llm=GeminiDecisionLLM(cfg), spanner_client=spanner_client, retriever=retriever
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'rules' or 'gemini'.")
    return gatekeepers


def run_comparison(
    gatekeepers: Dict[str, PaymentComplianceGatekeeper],
    dataset: List[Dict[str, Any]],
    repeats: int = 1,
    progress: bool = False,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Runs every case in every mode. Returns (raw runs, one scored row per run)."""
    runs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for mode, gk in gatekeepers.items():
        n_repeats = repeats if gk.use_vertex_llm else 1
        calls = getattr(gk.llm, "calls", None)
        for repeat in range(1, n_repeats + 1):
            for case in dataset:
                calls_before = len(calls) if calls is not None else 0
                output = ev.task_fn(case["input"], gk)
                new_calls = calls[calls_before:] if calls is not None else []
                runs.append({"mode": mode, "repeat": repeat, "case": case, "output": output})

                history = output["gate_history"]
                first_errors = history[0] if history else []
                row = {
                    "mode": mode,
                    "repeat": repeat,
                    "stratum": case.get("stratum", "core"),
                    "tx_id": case["input"]["tx_id"],
                    "expected": case["expected"]["decision"],
                    "got": output["decision"],
                    "gate_attempts": len(history),
                    "fell_back": not output["provenance_valid"],
                    "first_attempt_errors": " | ".join(first_errors),
                    "error_categories": ", ".join(sorted({categorise_error(e) for e in first_errors})),
                    "cited_rules": ", ".join(output["cited_rule_ids"]),
                    "cited_runbooks": ", ".join(output["cited_runbook_ids"]),
                    "retrieved_rules": ", ".join(output["retrieved_rule_ids"]),
                    "llm_calls": len(new_calls),
                    "llm_api_errors": sum(
                        1 for c in new_calls if c.get("error") and "did not parse" not in c["error"]
                    ),
                    "latency_s": round(sum(c.get("latency_s", 0.0) for c in new_calls), 2),
                    "graph_query_mode": output["graph_query_mode"],
                    "rationale": output["rationale"],
                }
                for scorer in ev.COMPARISON_SCORERS:
                    row[ev.metric_name(scorer)] = scorer({}, output, case["expected"])
                rows.append(row)
                if progress:
                    flag = "ok " if row["decision_match"] == 1.0 else "MISS"
                    print(f"  [{mode} #{repeat}] {row['tx_id']}: {row['got']:<17} "
                          f"expected {row['expected']:<17} {flag} "
                          f"gate attempts={row['gate_attempts']}"
                          f"{' FALLBACK' if row['fell_back'] else ''}")
    return runs, pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Counts and rates per group. Rates are over runs in the group."""
    records = []
    for keys, group in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(group)
        record = dict(zip(by, keys))
        record.update({
            "runs": n,
            "decision_correct": f"{int(group['decision_match'].sum())}/{n}",
            "false_release": int((group["no_false_release"] == 0.0).sum()),
            "over_restricted": int((group["no_over_restriction"] == 0.0).sum()),
            "docs_retrieved": f"{int(group['needed_documents_retrieved'].sum())}/{n}",
            "right_citations": f"{int(group['required_citations'].sum())}/{n}",
            "gate_first_pass": f"{int(group['gate_first_pass'].sum())}/{n}",
            "fell_back": int(group["fell_back"].sum()),
        })
        records.append(record)
    out = pd.DataFrame(records)
    if "stratum" in by:
        out["_order"] = out["stratum"].map({s: i for i, s in enumerate(STRATA_ORDER)}).fillna(99)
        out = out.sort_values(["_order"] + [b for b in by if b != "stratum"]).drop(columns="_order")
    return out.reset_index(drop=True)


def error_breakdown(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """mode -> how often each kind of error appeared in a first draft."""
    breakdown: Dict[str, Dict[str, int]] = {}
    for mode, group in df.groupby("mode", sort=False):
        counter: Counter = Counter()
        for cell in group["error_categories"]:
            for label in filter(None, (part.strip() for part in cell.split(","))):
                counter[label] += 1
        breakdown[mode] = dict(counter.most_common())
    return breakdown


def unstable_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Cases where repeats of the same mode did not all reach the same decision."""
    spread = df.groupby(["mode", "tx_id"])["got"].agg(lambda s: sorted(set(s)))
    spread = spread[spread.apply(len) > 1]
    return spread.reset_index().rename(columns={"got": "decisions_seen"})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modes", nargs="+", default=["rules"], choices=["rules", "gemini"])
    parser.add_argument("--repeats", type=int, default=3, help="runs per case for the Gemini path")
    parser.add_argument("--graph", default="mock", choices=["mock", "spanner"],
                        help="ownership graph backend (mock = local in-memory graph)")
    parser.add_argument("--embeddings", default="hash", choices=["hash", "gemini"],
                        help="vectors for FAISS retrieval (hash = offline keyword stand-in)")
    parser.add_argument("--dataset", default="all", choices=["core", "extended", "all"])
    parser.add_argument("--braintrust", action="store_true",
                        help="record the runs as Braintrust experiments (one per mode)")
    parser.add_argument("--vertex-eval", action="store_true",
                        help="submit the runs to the Vertex AI Gen AI evaluation service")
    parser.add_argument("--out", default="results", help="folder for result files")
    args = parser.parse_args(argv)

    dataset = {"core": ev.EVAL_DATASET, "extended": ev.EXTENDED_DATASET, "all": ev.ALL_CASES}[args.dataset]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print(f"Modes: {args.modes} | graph: {args.graph} | embeddings: {args.embeddings} | "
          f"cases: {len(dataset)} | Gemini repeats: {args.repeats}")
    gatekeepers = build_gatekeepers(args.modes, args.graph, args.embeddings)
    print("Running ...")
    runs, df = run_comparison(gatekeepers, dataset, repeats=args.repeats, progress=True)

    overall = summarise(df, ["mode"])
    by_stratum = summarise(df, ["stratum", "mode"])
    errors = error_breakdown(df)
    unstable = unstable_cases(df)

    print("\n=== Overall ===")
    print(overall.to_string(index=False))
    print("\n=== By case type ===")
    print(by_stratum.to_string(index=False))
    print("\n=== What the provenance gate rejected in first drafts ===")
    for mode, counts in errors.items():
        print(f"  {mode}: {counts if counts else 'nothing rejected'}")
    if len(unstable):
        print("\n=== Cases where repeated runs disagreed ===")
        print(unstable.to_string(index=False))

    misses = df[df["decision_match"] == 0.0]
    if len(misses):
        print("\n=== Decisions that did not match ===")
        print(misses[["mode", "repeat", "tx_id", "stratum", "expected", "got", "fell_back"]]
              .to_string(index=False))

    usage: Dict[str, Any] = {}
    for mode, gk in gatekeepers.items():
        if hasattr(gk.llm, "usage_summary"):
            usage[mode] = gk.llm.usage_summary()
    if usage:
        print("\n=== Gemini usage ===")
        for mode, u in usage.items():
            print(f"  {mode}: {u}")

    graph_modes = sorted(set(df["graph_query_mode"]))
    if "sql_fallback" in graph_modes:
        client = next(iter(gatekeepers.values())).spanner_client
        print("\nWARNING: the Spanner graph query failed and plain SQL was used instead:")
        print(f"  {client.last_query_error}")

    external: Dict[str, Any] = {}
    if args.braintrust:
        print("\n=== Braintrust ===")
        for mode in gatekeepers:
            mode_runs = [r for r in runs if r["mode"] == mode]
            try:
                external[f"braintrust_{mode}"] = ev.log_runs_to_braintrust(
                    mode_runs, mode, experiment_name=f"{mode}-{stamp}"
                )
                print(f"  {mode}: {external[f'braintrust_{mode}']}")
            except Exception as exc:  # noqa: BLE001 - reporting must not lose the results
                external[f"braintrust_{mode}"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {mode}: FAILED - {type(exc).__name__}: {exc}")
    if args.vertex_eval:
        print("\n=== Vertex AI Gen AI evaluation ===")
        for mode in gatekeepers:
            pairs = [(r["case"], r["output"]) for r in runs if r["mode"] == mode]
            try:
                result = ev.run_vertex_eval(
                    ev.eval_rows_to_dataframe(pairs), run_name=f"{mode}-{stamp}"
                )
                external[f"vertex_{mode}"] = result.summary_metrics
                print(f"  {mode}: {json.dumps(result.summary_metrics, default=str)}")
            except Exception as exc:  # noqa: BLE001
                external[f"vertex_{mode}"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {mode}: FAILED - {type(exc).__name__}: {str(exc)[:600]}")

    out_dir = Path(args.out) / f"comparison_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "runs.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "settings": vars(args),
        "overall": overall.to_dict(orient="records"),
        "by_stratum": by_stratum.to_dict(orient="records"),
        "first_draft_gate_errors": errors,
        "unstable_cases": unstable.to_dict(orient="records"),
        "gemini_usage": usage,
        "graph_query_modes": graph_modes,
        "external": external,
    }, indent=2, default=str), encoding="utf-8")
    (out_dir / "outputs.json").write_text(json.dumps([
        {"mode": r["mode"], "repeat": r["repeat"], "tx_id": r["case"]["input"]["tx_id"],
         "expected": r["case"]["expected"], "output": r["output"]}
        for r in runs
    ], indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'runs.csv'}, summary.json, outputs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
