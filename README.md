# Payment Compliance Gatekeeper

A LangGraph agent that decides whether a payment is blocked, escalated for review, or
released, based on beneficial-ownership tracing (the 50 percent rule), three further
evidence probes, and retrieved policy text. Every decision passes a deterministic
provenance gate before it is accepted.

The decision can be drafted two ways, and the point of this version is to compare them
on the same cases:

| Path | Who drafts the decision | Reads the remittance text |
|---|---|---|
| `rules` | deterministic if/else on the tool results | no |
| `gemini` | Gemini, with structured output | yes |

Everything else is shared: the tools, the retrieval, the gate, the retry and fallback
logic. A difference in outcome between the two paths is therefore down to the drafting
step alone.

All data is synthetic. Every entity, name and identifier is fictional, and the rule and
runbook texts are illustrative policy written for this demo, not statements of current law.

## What runs where

| Component | Offline default | Cloud backend | Switch |
|---|---|---|---|
| Decision drafting | rule path | Gemini (`gemini-3.8-flash`) | `--modes rules gemini` |
| Ownership graph | in-memory graph from `reference_data.py` | Spanner Graph, queried with GQL | `--graph spanner` |
| Retrieval vectors (index is FAISS in both) | hand-written keyword counts | Gemini embeddings (`gemini-embedding-001`) | `--embeddings gemini` |
| Experiment tracking | local scoring only | Braintrust | `--braintrust` |
| Trajectory metrics | local scorer | Vertex AI Gen AI evaluation | `--vertex-eval` |

The offline defaults are chosen in code, never by an environment variable, so the tests
stay offline whatever is set in your shell.

Still fixed responses, as in the original design: the Metrics, Traces and Logs probes.
Only the ownership trace is backed by a real data store.

## Files

| File | Purpose |
|---|---|
| `models.py` | data contracts and graph state |
| `compliance_graph_agent.py` | the LangGraph workflow: hypotheses, tool probing, retrieval, drafting, provenance gate, fallback |
| `spanner_graph_client.py` | ownership trace (local graph or Spanner Graph) and the three fixed probes |
| `reference_data.py` | entities and ownership edges, shared by the local graph and Spanner |
| `faiss_regulatory_store.py` | FAISS retrieval over rules and runbooks |
| `gemini_client.py` | Gemini adapters: decision drafting and embeddings |
| `braintrust_eval.py` | the 12 cases, the scorers, Braintrust and Vertex evaluation |
| `compare_modes.py` | runs both paths and prints the comparison |
| `check_setup.py` | preflight for the cloud pieces |
| `seed_spanner.py` | creates the Spanner schema, loads the graph, verifies it |
| `schema.sql` | Spanner tables and property graph |
| `test_compliance_gatekeeper.py` | the six original tests, unchanged |
| `test_extensions.py` | tests for everything added; all offline |

## Step 0 - offline (no account needed)

PowerShell, from the project folder. Python 3.11 or 3.12 is the safest choice.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

pytest -q
python braintrust_eval.py
python compare_modes.py
```

Expected: `58 passed, 2 skipped` (both skipped tests need the cloud packages; with
them installed it is `60 passed`), the three original decisions (BLOCK, MANUAL_ESCALATION, APPROVE), and a comparison table in
which the rule path gets 9 of 12 decisions right. The three it misses are the
`text_signal` cases: the only evidence is in the remittance text, which the rule path
does not read.

If `faiss-cpu` or `langgraph` fails to install on your Python version, remove it from
`requirements.txt` and continue. The code falls back to a NumPy index and a built-in
state machine with the same behaviour.

## Round 1 - Gemini, embeddings, Braintrust

Needs a Google Cloud project with billing and the Gemini platform (Vertex AI) API
enabled (`aiplatform.googleapis.com`), and the gcloud CLI.

```powershell
pip install -r requirements-cloud.txt

gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
$env:GOOGLE_CLOUD_PROJECT = "YOUR_PROJECT_ID"

python check_setup.py
python compare_modes.py --modes rules gemini --embeddings gemini --repeats 3
```

`check_setup.py` makes one Gemini call and a few embedding calls and prints PASS or FAIL
per item. Run the comparison once it is clean.

The comparison makes about 36 Gemini calls (12 cases, 3 repeats, plus retries) and
about 20 embedding calls on the first run; embeddings are cached in `.pcg_cache/`
afterwards. At list prices this is well under one US dollar.

To record the runs in Braintrust, create a free account, then:

```powershell
$env:BRAINTRUST_API_KEY = "sk-..."
python compare_modes.py --modes rules gemini --embeddings gemini --repeats 3 --braintrust
```

Without the key, `--braintrust` still runs and scores locally but sends nothing.

## Round 2 - Spanner Graph and Vertex evaluation

Create the Spanner instance only when you are ready for this round: a project gets one
free trial instance, and its 90 days start at creation.

```powershell
gcloud services enable spanner.googleapis.com --project YOUR_PROJECT_ID
gcloud spanner instances create pcg-trial --instance-type=free-instance `
    --config=regional-us-central1 --description="PCG trial" --project YOUR_PROJECT_ID

python seed_spanner.py
python check_setup.py --spanner
python compare_modes.py --modes rules gemini --embeddings gemini --graph spanner `
    --repeats 3 --braintrust --vertex-eval
```

`seed_spanner.py` creates the database and schema if they are missing, loads 24 entities
and 21 ownership edges, then compares Spanner's answer with the local graph for every
beneficiary. It ends with one of:

* `PASS` - the GQL query returns the same ownership paths as the local graph.
* `WARNING` - the GQL query failed and the tables were read with plain SQL instead. The
  message includes Spanner's error. The run continues, but Spanner Graph itself was not
  exercised; send the error back so the query can be corrected.
* `FAIL` - the two backends disagree.

`--vertex-eval` submits each mode's runs to the Gen AI evaluation service. The four
trajectory metrics are computed by the service; results appear under Vertex AI
Experiments as `payment-compliance-gatekeeper-eval`.

## The 12 cases

| Type | Cases | What decides the outcome |
|---|---|---|
| `core` | 3 | the original cases: 52% sanctioned ownership, a 30% offshore owner with a routing anomaly, a clean domestic payment |
| `structured` | 3 | ownership data only: 49.9% (review, not block), exactly 50% across two sanctioned parties, a three-hop chain with a cross-holding (56.7%) |
| `text_signal` | 3 | clean structured data; the remittance text names a sanctioned territory, asks to hide one, or mentions only a port |
| `text_distractor` | 3 | misleading text: place names used innocently, and instructions planted in the text on a payment that must be blocked or reviewed |

## Reading the comparison

| Column | Meaning |
|---|---|
| `decision_correct` | decision equals the expected one |
| `false_release` | approved a payment that should be held or blocked - the costly error |
| `over_restricted` | stricter than expected - safe, but it costs analyst time |
| `docs_retrieved` | retrieval returned the expected rule and runbook; independent of the path, and the ceiling for the next column |
| `right_citations` | the decision cites the expected rule and runbook |
| `gate_first_pass` | the first draft passed the provenance gate |
| `fell_back` | two drafts were rejected and the deterministic escalation was used |

The report also lists what the gate rejected in first drafts (wrong citation, ownership
arithmetic, and so on), cases where repeated Gemini runs disagreed, and token use with an
estimated cost.

The rule path is deterministic, so its row is the same every time. For Gemini, treat
the numbers as measurements of this model on these twelve cases, not as general results.

## Configuration

| Variable | Default | Use |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | none | required for any cloud backend |
| `PCG_GEMINI_MODEL` | `gemini-3.8-flash` | decision model |
| `PCG_GEMINI_LOCATION` | `global` | this model is served from `global`, `us`, `eu`; not from `us-central1` |
| `PCG_THINKING_LEVEL` | `LOW` | `LOW`, `MEDIUM`, `HIGH`, or `DEFAULT` to leave it unset |
| `PCG_TEMPERATURE` | unset | left at the model default unless set |
| `PCG_EMBED_MODEL` | `gemini-embedding-001` | embedding model |
| `PCG_EMBED_LOCATION` | `us-central1` | embedding endpoint |
| `PCG_EMBED_DIM` | `768` | embedding size |
| `PCG_VERTEX_LOCATION` | `us-central1` | Gen AI evaluation service |
| `PCG_SPANNER_INSTANCE` | `pcg-trial` | Spanner instance id |
| `PCG_SPANNER_DATABASE` | `compliance` | Spanner database id |
| `PCG_PRICE_IN_PER_M`, `PCG_PRICE_OUT_PER_M` | `0.75`, `3.75` | USD per million tokens, for the cost estimate only |
| `BRAINTRUST_API_KEY` | none | send experiments to Braintrust |

## Changes from the original package

* **Ownership data** has one source (`reference_data.py`) for both backends. The original
  sample recorded 128% direct ownership of one company; holdings now sum to at most 100%
  while the sanctioned total stays 52%. The local backend traverses the graph instead of
  returning fixed paths.
* **Spanner query** rewritten for Spanner Graph (`ACYCLIC`, `ARRAY_TRANSFORM` over
  `NODES`/`EDGES`); the original used a list comprehension Spanner does not accept.
  Intermediate holders are filtered so only the end of each ownership chain is counted.
* **Gemini** is called through the `google-genai` SDK with a current model. The original
  model id, `gemini-2.5-pro`, has been retired on this endpoint.
* **Retrieval** can use real embeddings, and a second retrieval is driven by the
  remittance text.
* **Rule path** fixes: its rationale is built from the transaction's own findings (it
  was a fixed sentence that described the first sample case), and a sanctioned stake
  below 50% now retrieves the due-diligence documents rather than the freeze runbook.
* **Evaluation**: nine more cases, Braintrust is actually called, the Vertex evaluation
  initialises its client and sends trajectories in the format the service expects.
* `langchain-google-vertexai` is no longer a dependency.

## Known limitations

* **The gate checks the decision itself in one case only**: sanctioned ownership of 50%
  or more must be BLOCK. Below that it checks grounding (citations, evidence ids,
  arithmetic), not whether the decision is right. A well-cited APPROVE on a payment that
  needs review passes. This is the original design, kept as is; the `false_release`
  column is where it shows. `test_extensions.py` has a test that documents it.
* **Tool selection is done by code.** Gemini drafts the decision after the tools have
  run; it does not choose them.
* **Three of the four probes are fixed responses** keyed by beneficiary.
* **Twelve cases** are enough to see how each path behaves, not to estimate rates.
* **Offline retrieval is not semantic.** With the keyword vectors, the text-driven
  search returns an arbitrary neighbour. Use `--embeddings gemini` for it to mean anything.
* **Cloud paths verified on 2026-11-14** against live services: Gemini, Gemini embeddings, Spanner Graph and the Vertex AI evaluation service. Braintrust was run in local mode only.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Gemini call: 404, model not found | wrong location; keep `PCG_GEMINI_LOCATION=global` |
| 403 or "API not enabled" | enable `aiplatform.googleapis.com`; check the project id |
| 429 | quota; trial accounts have low limits for generative models. Lower `--repeats` |
| `DefaultCredentialsError` | run `gcloud auth application-default login` |
| embedding returned a different dimension | set `PCG_EMBED_DIM` to a size the model supports (768, 1536, 3072) |
| `seed_spanner.py`: instance not found | create the free trial instance, or set `PCG_SPANNER_INSTANCE` |
| `seed_spanner.py` ends in WARNING | the GQL query was rejected; the error text is printed above it |
| many `fell_back` with "no usable draft" | the model calls are failing; see `llm_api_errors` in `runs.csv` |

## Clean-up

Disabling an API does not remove resources. Delete the Spanner instance when you are
done (`gcloud spanner instances delete pcg-trial`), or let the trial expire without
upgrading it. Gemini and embedding calls bill only when made.

