# Academy Co-Scientist — Architecture & Design

## Overview

The Academy Co-Scientist is an **autonomous multi-agent research platform** built on the [Academy](https://github.com/proxystore/academy) agent framework. It orchestrates a pipeline of specialized agents to autonomously generate, review, refine, rank, and validate scientific hypotheses for a given research topic.

The entire system is driven by a YAML configuration file with no Python changes needed to tune agent counts, models, timeouts, or backends.

---

## Entry Point

```
python -m academy_coscientist.simulator --config configs/simulator_config.yaml
```

The `simulator.py` module is the single entry point. `launcher.py` has been removed.

---

## Agent Inventory

| Agent | File | Role |
|---|---|---|
| `HypothesisGenerationAgent` | `agents/generation_agent.py` | Brainstorm candidate hypotheses using an LLM; RAG context from `query_texts`; deduplicate globally |
| `ReviewAgent` | `agents/review_agent.py` | Score hypotheses on 5 dimensions (novelty, rigor, feasibility, impact, clarity) and push scores |
| `TournamentAgent` | `agents/tournament_agent.py` | Central leaderboard — stores hypotheses, ingests scores, serves sorted rankings |
| `HypothesisRefinerAgent` | `agents/refiner_agent.py` | Rewrite a hypothesis to address reviewer critique; receives full review history **and per-dimension scores** (novelty, rigor, feasibility, impact, clarity); explicitly targets low-scoring dimensions — especially novelty |
| `TeamCoordinatorAgent` | `agents/team_coordinator_agent.py` | Orchestrate one team: generate → review → refine loop until threshold or max rounds |
| `MetaReviewAgent` | `agents/meta_agent.py` | Produce literature-grounded meta-reviews by combining the leaderboard with retrieved papers |
| `ResearchVectorDBAgent` | `agents/research_vector_agent.py` | FAISS-based local vector store for research paper abstracts |
| `PaperHarvesterAgent` | `agents/paper_harvester_agent.py` | Discover and download open-access papers (S2 → OpenAlex → arXiv), rebuild vector index |
| `ProofOfConceptAgent` | `agents/poc_agent.py` | Orchestrate PoC pipeline: generate code → delegate execution → interpret results → save report |
| `DockerExecutorAgent` | `agents/docker_executor_agent.py` | Isolated code execution sandbox: run Python code in a fresh Docker container via stdin; return stdout/stderr/returncode |
| `ProvenancePDFReportAgent` | `agents/provenance_report_agent.py` | Render a comprehensive multi-section PDF report with provenance statistics |

---

## Execution Flow

### 1. Startup (`simulator.main`)

```
CLI args → load YAML config → apply CLI overrides
         → init_run_context()         # create runs/<timestamp-uuid>/
         → load_config()              # register models/paths
         → resolve output paths       # flowcept, poc, report → runs/<id>/
         → start FlowCept plugin      # optional provenance capture
         → asyncio.run(_run_simulator)
```

### 2. Agent Wiring (`_run_simulator`)

```
Build Academy exchange (local / Redis / HTTP)
Build executor (thread pool / process pool)
Open Manager context
│
├─ Launch ResearchVectorDBAgent        [shared, optional]
├─ Launch PaperHarvesterAgent(s)       [background, weekly harvest]
├─ Launch TournamentAgent              [global leaderboard]
├─ Launch MetaReviewAgent(s)           [periodic synthesis]
├─ Launch ProvenancePDFReportAgent     [triggered on shutdown]
│
├─ IF teams configured → TEAM MODE
│   └─ _launch_teams(...)             # also launches DockerExecutorAgent + PoC agent
│
└─ ELSE → FLAT MODE
    └─ _launch_generation_agents + _launch_review_agents
```

### 3. Team Mode (recommended)

Teams are chained pipelines. Each team runs an independent review-refine loop and hands validated hypotheses downstream.

#### Rules (apply to any number of teams)

| Rule | Detail |
|---|---|
| **Only Team 0 generates** | `HypothesisGenerationAgent` runs only in Team 0; all other teams are pure refiners that pull from their incoming staging tournament |
| **Staging chain** | N teams → N−1 staging `TournamentAgent`s; Team i pushes validated hypotheses into `staging[i]`, which Team i+1 reads from |
| **All teams push to global** | Every team pushes its validated hypotheses directly to the shared global `TournamentAgent`, not just the final team — the leaderboard is populated progressively |
| **Threshold escalation** | Each team's pass threshold = `score_threshold + team_idx × 0.10`, capped at 0.95; Team 0 is the most lenient, Team N−1 the strictest |
| **Discard path** | A hypothesis below `min_pass_score` after `max_refinement_rounds` is silently dropped and never reaches the global tournament |

#### Pipeline with N teams

```
                             global TournamentAgent  ◄──── all teams write here
                                     ▲  ▲  ▲
                                     │  │  │
Team 0  LLM generate + refine  ──────┘  │  │
        threshold = score_threshold      │  │
        validated → staging[0] ──────────┘  │
                                            │
Team 1  pull staging[0] + refine  ──────────┘
        threshold = score_threshold + 0.10
        validated → staging[1] ──────────────► Team 2 …
…
Team N-1  pull staging[N-2] + refine
          threshold = min(score_threshold + (N-1)×0.10, 0.95)
          validated → global TournamentAgent only

After all teams finish:
  ProofOfConceptAgent runs on top-N hypotheses from global leaderboard
    └─ calls DockerExecutorAgent.run_code(code) for each attempt
         └─ docker run --rm -i --network none --memory 512m image python -
    └─ interprets results → verdict: SUPPORTED | PARTIALLY_SUPPORTED |
                                     INCONCLUSIVE | REFUTED
    └─ IF verdict == INCONCLUSIVE:
         → embeds PoC evidence in hypothesis meta.poc_feedback
         → calls last team's resubmit_with_poc_feedback(hyp, poc_feedback)
         → team re-runs the full review-refine loop with experimental context
```

#### Example: 3 teams, `score_threshold: 0.75`

```
Team 0  generates from LLM   threshold 0.75  → staging[0] + global
Team 1  refines staging[0]   threshold 0.85  → staging[1] + global
Team 2  refines staging[1]   threshold 0.95  → global only
```

Setting `count: 4` would add a Team 3 at threshold 0.95 (capped) pulling from `staging[2]`.

#### Review-Refine Loop (per hypothesis, per team)

```
For each hypothesis:
  1. All ReviewAgents call review_one(hyp_id, hyp)
     → Each reviewer scores on 5 dimensions: novelty, rigor, feasibility,
       impact, clarity (each 1-10) + produces reasoning, weaknesses, risks
  2. Critiques merged: avg composite score, per-dimension averages,
     deduplicated weaknesses, all strengths, majority-vote recommendation
  3. IF avg_score >= team_threshold  → VALIDATED, push to outgoing tournament
  4. ELSE IF round < max_rounds      → refine(hyp, merged_critique, history)
       The refiner receives:
         - Merged critique with ALL dimension scores
         - ⚠️ Flag for dimensions scoring < 6/10 (priority targets)
         - Special instruction to improve NOVELTY if novelty < 6
         - Full history of all prior rounds (score trajectory, recurring issues)
       → go to step 1 with improved hypothesis
  5. ELSE IF avg_score >= min_pass   → push anyway (partial quality)
  6. ELSE                            → discard
```

The full review history (all rounds) is passed to the refiner on every call so recurring weaknesses are flagged with ⚠️. Dimensions scoring below 6/10 are explicitly flagged; the refiner is instructed to introduce a genuinely new angle or mechanism when novelty is low.

### 4. Flat Mode (legacy)

When no `teams:` section is configured, standalone generation and review agents loop autonomously:

- `HypothesisGenerationAgent.generation_loop()` — propose N hypotheses every interval, push to tournament
- `ReviewAgent.review_loop()` — review all hypotheses in tournament, push scores back

No refinement loop is performed in flat mode.

### 5. Shutdown Sequence

```
SIGINT / SIGTERM / max_duration / PoC completion
  → shutdown_event.set()
  → cancel status_loop task
  → _generate_pdf_report()   # flush provenance buffer, render PDF
  → flowcept_plugin.stop()   # finalise perf CSV
```

---

## Data Flow

```
  PaperHarvesterAgent (background)
  S2 → OpenAlex → arXiv
         │ rebuild_index
         ▼
  ResearchVectorDBAgent (FAISS)
    ├─── query_texts (RAG) ──────────────────────────────┐
    │                                                    │
    │ RAG context                                        ▼
    ▼                                         ┌──────────────────────────┐
  ┌───────────────────────────────────┐       │  MetaReviewAgent         │
  │  Team 0  (TeamCoordinatorAgent)   │       │  (periodic loop)         │
  │                                   │       │                          │
  │  HypothesisGenerationAgent        │       │  get_leaderboard() ◄──┐  │
  │  LLM brainstorm + RAG dedup       │       │  fetch literature     │  │
  │       │ propose_hypotheses(n)     │       │  write_meta_review    │  │
  │       ▼                           │       └──────────────────────────┘
  │  ┌─────────────────────────────┐  │                               │
  │  │  Review-Refine Loop         │  │                               │
  │  │                             │  │                               │
  │  │  ReviewAgent × N            │  │                               │
  │  │    novelty / rigor /        │  │                               │
  │  │    feasibility / impact /   │  │                               │
  │  │    clarity + reasoning CoT  │  │                               │
  │  │       │ score < threshold   │  │                               │
  │  │       ▼                     │  │                               │
  │  │  HypothesisRefinerAgent     │  │                               │
  │  │    all dimension scores     │  │                               │
  │  │    ⚠️ dims < 6/10 flagged   │  │                               │
  │  │    novelty targeted if low  │  │                               │
  │  │    full review history      │  │                               │
  │  │       │ improved hyp        │  │                               │
  │  └───────┘ (retry loop)        │  │                               │
  │       │ validated              │  │                               │
  └───────┼────────────────────────┘  │                               │
          │  Team 1 pulls staging[0]  │                               │
          │  Team 2 pulls staging[1]… │                               │
          ▼                           │                               │
  ┌────────────────────┐              │                               │
  │  TournamentAgent   │◄─ all teams push validated hyps              │
  │  global leaderboard│──────────────────────────────────────────────┘
  └────────┬───────────┘
           │ get_leaderboard(top_N)
           ▼
  ┌────────────────────────────────────────────────────────┐
  │  ProofOfConceptAgent                                   │
  │                                                        │
  │  1. generate_poc_code (Claude)                         │
  │  2. DockerExecutorAgent.run_code(code)                 │
  │       docker run --rm -i --network none python -       │
  │       → stdout / stderr / returncode                   │
  │  3. interpret_poc_results → verdict + confidence       │
  │  4. _save_report → poc_results/<id>/                   │
  │       poc_code.py  report.json  Dockerfile  run.sh     │
  └───────┬─────────────────────────┬──────────────────────┘
          │ SUPPORTED /             │ INCONCLUSIVE
          │ PARTIALLY_SUPPORTED /   ▼
          │ REFUTED       TeamCoordinatorAgent
          │               resubmit_with_poc_feedback()
          │               embeds evidence in meta.poc_feedback
          │               re-runs review-refine loop with context
          │               validated hyp → TournamentAgent
          │
          ▼
  ┌────────────────────────────────────────────────────────────┐
  │  ProvenancePDFReportAgent  (reads only, never writes to    │
  │                             provenance files)              │
  │                                                            │
  │  reads: actions.jsonl    ← reasoning, reviews,            │
  │         llm_calls.jsonl    refinements, brainstorm CoT     │
  │         provenance_perf.csv                                │
  │         TournamentAgent  ← top hypotheses                  │
  │         MetaReviewAgent  ← last meta-review summary        │
  │                                                            │
  │  → report_{date}.pdf                                       │
  └────────────────────────────────────────────────────────────┘
```

---

## Output Directory Layout

Every run creates an isolated directory under `runs/`:

```
runs/
└── 20260310-031053-<uuid>/
    ├── events.log               # human-readable log
    ├── actions.jsonl            # structured agent action log
    ├── llm_calls.jsonl          # every LLM prompt + response
    ├── provenance_buffer.jsonl  # FlowCept workflow records
    ├── provenance_perf.csv      # FlowCept overhead timings
    ├── poc_results/
    │   ├── poc_<id>_code.py
    │   └── poc_<id>_report.json
    └── report_<date>.pdf
```

Paths default to the run directory; explicit non-null values in config take precedence.

---

## Paper Harvesting & Literature Retrieval

Papers are fetched through a three-source fallback chain:

| Priority | Source | Notes |
|---|---|---|
| 1 | **Semantic Scholar** | Most-cited, best metadata; requires `SEMANTIC_SCHOLAR_API_KEY` (optional) |
| 2 | **OpenAlex** | Fully open; abstracts reconstructed from inverted index; set `OPENALEX_EMAIL` for polite pool |
| 3 | **arXiv** | Best for newest preprints; parsed from Atom XML feed |

The `MetaReviewAgent` fetches literature at review time using the same chain via `utils/utils_papers.py`, or from the local vector DB if populated.

---

## Provenance & Observability

Every agent action and LLM call is recorded:

- **`actions.jsonl`** — structured JSON per `log_action(logger, action, input, output)` call.
  Key fields captured per action type:
  | Action | Key provenance fields |
  |---|---|
  | `review_single` | `score`, `confidence`, `recommendation`, **`reasoning`** (full reviewer chain-of-thought) |
  | `refine` | `round`, `score_before`, **`reasoning`** (refiner analysis), **`revision_notes`** (what changed) |
  | `brainstorm_reasoning` | `topic`, `attempt`, **`reasoning`** (LLM chain-of-thought before proposing hypotheses) |
  | `poc_complete` | `verdict`, `confidence` |
  | `poc_feedback_sent` | `hyp_id`, `status` (emitted when INCONCLUSIVE result is resubmitted) |

- **`llm_calls.jsonl`** — full LLM prompt/response via `record_llm_call(payload)`.
  Every JSON-producing LLM call (`_call_llm_json`) surfaces `reasoning` as a **top-level key** alongside `parsed_response`, so the chain-of-thought is always directly accessible without traversing nested objects.

- **FlowCept plugin** (optional) — captures provenance as workflow/task records with overhead timing; enable with `flowcept.enabled: true` in config

The `ProvenancePDFReportAgent` **only reads** `actions.jsonl`, `llm_calls.jsonl`, and the provenance CSV to generate the PDF report. It does not modify or append to any provenance file.

### Provenance Integrity Constraint

**Raw provenance content must never be passed through an LLM.** This means the report agent's internal LLM calls (for generating the executive summary and per-hypothesis narrative paragraphs) receive only structured numerical data — scores, counts, verdicts, titles, token totals — never the free-text content stored in provenance:

| Provenance field | Handling in PDF |
|---|---|
| Reviewer `reasoning` text | Rendered **verbatim** in the per-hypothesis appendix section |
| Refiner `revision_notes` | Rendered **verbatim** in the refinement history section |
| PoC `interpretation` text | Rendered **verbatim** in the PoC details section |
| MetaReview `summary` text | Rendered **verbatim** in the Meta-Review Commentary section |
| Scores, counts, verdicts | Passed to LLM for narrative generation (safe — non-interpretive) |

LLMs are only permitted to generate connecting narrative prose from structured numerical inputs. Any free-text reasoning or interpretation recorded in provenance is treated as a primary source and reproduced as-is.

---

## Configuration Reference

All configuration lives in a single YAML file (see `configs/simulator_config.yaml`). Top-level sections:

| Section | Purpose |
|---|---|
| `models` | LLM model IDs by role: `reasoning`, `writing`, `embedding`, `poc`, `report` |
| `simulator` | Agent counts, topics, intervals, exchange type, executor, team settings |
| `flowcept` | Provenance plugin toggle and output paths |
| `poc` | Proof-of-concept agent: `enabled`, `top_n`, `max_retries`, `output_dir`, `docker_image`, `code_timeout`, `docker_memory`, `docker_network` |
| `report` | PDF report: `enabled`, `top_n`, `output_dir`, `output_filename` |

All `*_dir` and `*_path` fields default to `null`, which auto-resolves to a subdirectory inside the current run's output directory.

---

## Docker Execution Sandbox

`DockerExecutorAgent` isolates PoC code from the host:

| Parameter | Config key | Default | Notes |
|---|---|---|---|
| Image | `poc.docker_image` | `python:3.11-slim` | Any image with a `python` binary |
| Timeout | `poc.code_timeout` | `90` s | Wall-clock limit; returns timeout error beyond this |
| Memory | `poc.docker_memory` | `512m` | Docker `--memory` flag |
| Network | `poc.docker_network` | `none` | `none` = fully air-gapped; `bridge` if script needs internet |

Code is fed via **stdin** (`python -`) — no host filesystem mounts, no temp files, no shell-escaping risks.

If Docker is not available (daemon unreachable), the PoC agent logs a warning and execution returns a non-zero code; interpretation still runs on the empty output.

---

## Utilities

| Module | Purpose |
|---|---|
| `utils/config.py` | Global YAML config loader; `get_model(role)`, `get_temperature(role)` — roles include `writing`, `commentary`, `summary`, `meta_review` (0.1), `reasoning` |
| `utils/utils_logging.py` | Run context init, structured JSONL logging, plugin hook registry |
| `utils/utils_llm.py` | All LLM calls (OpenAI + Anthropic), embeddings, PoC codegen/interpret |
| `utils/utils_papers.py` | Shared paper search utility (S2 → OpenAlex → arXiv); used by MetaReviewAgent and PaperHarvesterAgent |
| `plugins/flowcept_plugin.py` | Optional FlowCept provenance plugin wired via `register_action_hook` |
