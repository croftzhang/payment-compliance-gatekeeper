# Payment Compliance Gatekeeper

A LangGraph agent that decides whether a payment is blocked, escalated for review, or
released. An investigation answers two questions that sanctions work actually turns on:

1. **Is the beneficiary itself a listed party?** The name is screened against a
   sanctions list; each hit is then settled by comparing identifiers (name
   disambiguation).
2. **Is the beneficiary owned 50 percent or more by listed parties?** Ownership is
   traced through intermediate holders, across several paths, and summed.

Three further evidence probes and retrieved policy text feed the decision, and every
decision passes a deterministic provenance gate before it is accepted.

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
| Sanctions list and name matching | local, in `reference_data.py` and `name_screening.py` | same (not a cloud service) | none |
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
| `reference_data.py` | entities, ownership edges and the sanctions list |
| `name_screening.py` | name matching against the list, and adjudication of the hits |
| `faiss_regulatory_store.py` | FAISS retrieval over rules and runbooks |
| `gemini_client.py` | Gemini adapters: decision drafting and embeddings |
| `braintrust_eval.py` | the 18 cases, the scorers, Braintrust and Vertex evaluation |
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

Expected: `76 passed, 3 skipped` (the skipped tests need the cloud packages; with them
installed it is `79 passed`), the three original decisions (BLOCK, MANUAL_ESCALATION,
APPROVE), and a comparison table in which the rule path gets 14 of 18 decisions right.
The four it misses are all approvals of payments that needed review: the three
`text_signal` cases, where the only evidence is in remittance text the rule path does
not read, and one `name_near_miss` case, where a listed name is abbreviated enough to
score just under the screening threshold.

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

The comparison makes about 54 Gemini calls (18 cases, 3 repeats, plus retries) and
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

## Name screening

Screening and adjudication are separate steps, as they are in practice.

**Screening** runs first on every payment. The beneficiary name is compared with every
listed name and alias after normalisation (case, punctuation and legal-form words such
as Ltd, LLC, FZE are ignored; word order does not matter). The score is the standard
library's sequence ratio, between 0 and 1.

| Score | Level | What happens |
|---|---|---|
| 0.85 or more | `ALERT` | opens the hypothesis `HYP-LIST-MATCH`; the adjudication tool is called first |
| 0.70 to 0.85 | `NEAR_MISS` | code takes no action; the entry is shown to Gemini, with the list-match documents retrieved for it |
| below 0.70 | none | not reported |

**Adjudication** (`resolve_list_hits`) compares the country and registration number on
the payment with the list entry:

| Finding | Verdict | Effect |
|---|---|---|
| registration numbers match | `TRUE_MATCH` | BLOCK, enforced by the gate |
| identical name, same country, no conflicting number | `TRUE_MATCH` | BLOCK, enforced by the gate |
| registration number and country both differ | `FALSE_POSITIVE` | no reason to hold; the reason is recorded |
| anything else | `UNRESOLVED` | rule path escalates; Gemini may settle it either way |

When the payment carries no beneficiary name, the name and country on the beneficiary's
own record are used, so a payment made straight to a listed entity is caught even
though ownership tracing, which looks at who owns the beneficiary, finds nothing.

The list has three synthetic entries and lives in `reference_data.py`. The matching is
local code in both modes; it is not one of the cloud services.

## The 18 cases

| Type | Cases | What decides the outcome |
|---|---|---|
| `core` | 3 | the original cases: 52% sanctioned ownership, a 30% offshore owner with a routing anomaly, a clean domestic payment |
| `structured` | 3 | ownership data only: 49.9% (review, not block), exactly 50% across two sanctioned parties, a three-hop chain with a cross-holding (56.7%) |
| `text_signal` | 3 | clean structured data; the remittance text names a sanctioned territory, asks to hide one, or mentions only a port |
| `text_distractor` | 3 | misleading text: place names used innocently, and instructions planted in the text on a payment that must be blocked or reviewed |
| `name_alert` | 4 | the name scores as an alert: a listed party confirmed by registration number, one confirmed by identical name and country, a different company with the same name, and a transliterated spelling that cannot be settled |
| `name_near_miss` | 2 | two names that both score 0.80: an abbreviated form of a listed name, and an unrelated company sharing two common words |

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
the numbers as measurements of this model on these eighteen cases, not as general results.

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
* **Name screening** is new: a synthetic sanctions list, a screening step that runs
  before the hypotheses are formed, an adjudication tool, two gate checks, and six cases.
* **Evaluation**: fifteen more cases in all, Braintrust is actually called, the Vertex evaluation
  initialises its client and sends trajectories in the format the service expects.
* `langchain-google-vertexai` is no longer a dependency.

## Known limitations

* **The gate checks the decision itself in one case only**: sanctioned ownership of 50%
  or more must be BLOCK. Below that it checks grounding (citations, evidence ids,
  arithmetic), not whether the decision is right. A well-cited APPROVE on a payment that
  needs review passes. This is the original design, kept as is; the `false_release`
  column is where it shows. `test_extensions.py` has a test that documents it.
* **The same holds for unresolved name alerts.** Settling a hit that code could not is
  the model's job, so a clearance it gets wrong is a false release the gate does not stop.
  Only a `TRUE_MATCH` verdict is binding.
* **Name matching is deliberately simple**: one similarity measure, two thresholds, two
  identifiers, a three-entry list. It screens the beneficiary name only, not other
  parties and not names inside the remittance text.
* **Tool selection is done by code.** Gemini drafts the decision after the tools have
  run; it does not choose them.
* **Three of the four probes are fixed responses** keyed by beneficiary.
* **Eighteen cases** are enough to see how each path behaves, not to estimate rates.
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

