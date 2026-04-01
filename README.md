# Academy Co-Scientist

**Academy Co-Scientist** is an autonomous multi-agent research assistant pipeline built on [Academy](https://github.com/academy-agents/academy). Agents collaborate to generate, review, refine, and experimentally validate scientific hypotheses — entirely autonomously.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Agents Overview](#agents-overview)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Multi-Framework Provenance](#multi-framework-provenance)
- [Project Structure](#project-structure)
- [Distributed Execution](#distributed-execution)

---

## Features

- **Team-based co-scientist pipeline** — chained teams of Generator + Refiner + Reviewers, each operating at progressively stricter quality thresholds
- **Hypothesis deduplication** — three-layer dedup: shared in-process registry across all agents and cycles, tournament-level title index, and LLM retry loop
- **RAG-augmented generation** — hypotheses informed by relevant paper abstracts retrieved from a FAISS vector store via `query_texts`
- **Autonomous paper harvesting** — downloads and indexes open-access papers from Semantic Scholar, OpenAlex, and arXiv on a configurable schedule
- **Iterative review-refine loop** — hypotheses refined up to `max_refinement_rounds` times; the refiner receives full review history, all per-dimension scores (novelty, rigor, feasibility, impact, clarity), and explicit instructions to target low-scoring dimensions — especially novelty
- **INCONCLUSIVE PoC feedback** — when a proof-of-concept returns an inconclusive verdict, the hypothesis is automatically resubmitted to the last team's review-refine loop with the experimental evidence embedded as context
- **Proof-of-concept execution** — top-N hypotheses are turned into runnable Python experiments by Claude, executed, and interpreted; broken scripts are fixed and retried
- **Coherent pipeline shutdown** — simulator auto-stops once the PoC agent finishes; all agents respect `max_cycles`
- **Full provenance capture** — every agent action and LLM call is recorded; reviewer reasoning, refiner chain-of-thought, and brainstorm reasoning are all surfaced as first-class fields in `actions.jsonl` and `llm_calls.jsonl` — the `ProvenancePDFReportAgent` reads these files and generates a PDF report without modifying them
- **Multi-framework provenance** — FlowCept plugins for Academy, LangGraph, CrewAI, and AutoGen share a single buffer; all records carry the same `campaign_id` and `workflow_id` for end-to-end lineage across frameworks. CrewAI capture uses native LLM/tool hooks for rich per-call provenance (full message lists, agent context, iteration counts)
- **Flexible backends** — local threads, Redis, or Globus/HTTP exchange; thread or process executors
- **Config-driven** — everything tunable via a single YAML file; CLI overrides for quick experiments

---

## Architecture

```
                          ┌─────────────────────────────┐
                          │     PaperHarvesterAgent      │
                          │  S2 → OpenAlex → arXiv       │
                          └──────────────┬───────────────┘
                                         │ rebuild_index()
                          ┌──────────────▼───────────────┐
                          │     ResearchVectorDBAgent     │
                          │       (FAISS + embeddings)    │
                          └──────────────┬───────────────┘
                                         │ query_texts() — RAG context
              ┌──────────────────────────▼──────────────────────────┐
              │                       TEAM 0                         │
              │  HypothesisGenerationAgent  (brainstorms N ideas)    │
              │        │                                             │
              │  ReviewAgent × N  (score + critique each idea)      │
              │        │                                             │
              │  HypothesisRefinerAgent  (improve on full history)   │
              │        │                                             │
              │  TeamCoordinatorAgent  (orchestrates the cycle)      │
              └────────────────┬─────────────────────────────────────┘
                               │ validated hypotheses (score ≥ threshold)
              ┌────────────────▼─────────────────────────────────────┐
              │                       TEAM 1                         │
              │  (pulls from staging, refines to higher standard)    │
              └────────────────┬─────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   TournamentAgent   │
                    │  (global ranking)   │
                    └──────────┬──────────┘
                               │ top-N hypotheses
                    ┌──────────▼──────────┐
                    │  ProofOfConceptAgent │
                    │  generates code →   │
                    │  executes → retries │
                    │  → interprets       │
                    └─────────────────────┘
                          poc_results/
```

**Team chaining**: Team 0 generates from scratch and forwards hypotheses that pass its threshold to a staging tournament. Team 1 reads the staging tournament and refines further at a 0.10-higher threshold before pushing to the global tournament. Additional teams can be added by increasing `teams.count`.

**PoC agent**: After all teams complete `max_cycles`, the PoC agent takes the top-N hypotheses from the global tournament, asks Claude to write a real Python experiment for each, runs it in a subprocess, fixes errors (up to `max_retries` times), and saves code + interpretation to `poc_results/`.

---

## Agents Overview

| Agent | Role |
|---|---|
| `HypothesisGenerationAgent` | Brainstorms diverse, falsifiable hypotheses; RAG-augmented via `query_texts` |
| `HypothesisRefinerAgent` | Rewrites a hypothesis given reviewer critique and full review history; receives all per-dimension scores and explicitly targets low-scoring dimensions, especially novelty |
| `ReviewAgent` | Scores a hypothesis on novelty, rigor, feasibility, impact, clarity |
| `TeamCoordinatorAgent` | Runs the generate → review → refine loop for one team |
| `TournamentAgent` | Maintains the global ranked leaderboard; deduplicates by title |
| `PaperHarvesterAgent` | Downloads open-access papers (S2 → OpenAlex → arXiv); rebuilds vector index |
| `ResearchVectorDBAgent` | FAISS vector store; serves abstract text via `query_texts` for RAG context |
| `ProofOfConceptAgent` | Orchestrates PoC pipeline: generate code → delegate execution → interpret results |
| `DockerExecutorAgent` | Isolated sandbox: runs Python code via stdin in a fresh Docker container; returns stdout/stderr/returncode |
| `ProvenancePDFReportAgent` | Renders a multi-section PDF report with provenance statistics on shutdown |

---

## Setup

### Requirements

- Python 3.11+
- **Docker** — required for PoC code execution (the `docker` CLI must be on `PATH` and the daemon running)
- **LLM credentials** — one of:
  - **Argo gateway** (ANL): set `ARGO_USER` env var or `argo_user:` in the config; no other keys needed
  - **Direct APIs**: `OPENAI_API_KEY` for generation/review/refinement, `ANTHROPIC_API_KEY` for PoC and report

### Install

```bash
git clone https://github.com/YOUR_USERNAME/academy-coscientist.git
cd academy-coscientist
python -m venv venv
source venv/bin/activate
pip install -e .
```

### API Keys

**Standard mode** (direct OpenAI / Anthropic):

```bash
export OPENAI_API_KEY="sk-..."         # required: generation, review, refinement
export ANTHROPIC_API_KEY="sk-ant-..."  # required: PoC code generation, PDF report narratives
```

**Argo mode** (ANL gateway — replaces both keys above):

```bash
export ARGO_USER="your_anl_username"   # set this OR add argo_user: in the config
```

**Paper harvester rate limits** (optional):

```bash
export SEMANTIC_SCHOLAR_API_KEY="..."   # raises S2 from 1 req/s to 10 req/s
export OPENALEX_EMAIL="you@example.com" # joins OpenAlex polite pool (higher limits)
```

### Python Dependencies

| Package | Purpose |
|---|---|
| `academy-py` | Multi-agent framework (Agent, Manager, exchange, executors) |
| `openai>=1.40` | Generation, review, meta-review LLM calls |
| `anthropic>=0.40` | PoC code generation and interpretation (Claude) |
| `faiss-cpu>=1.8` | Vector similarity search for RAG |
| `sentence-transformers>=3.0` | Local embedding model for FAISS |
| `pypdf>=4.0` | PDF text extraction — harvested papers and PDF-as-topic |
| `pyyaml>=6.0` | YAML config loading |
| `reportlab` | PDF report rendering |
| `flowcept` _(optional)_ | Provenance tracking |
| `langgraph>=0.2` _(optional)_ | LangGraph provenance plugin |
| `langchain-core>=0.2` _(optional)_ | Required by LangGraph plugin |
| `crewai>=1.0` _(optional)_ | CrewAI provenance plugin |
| `autogen-agentchat>=0.4` _(optional)_ | AutoGen provenance plugin |
| `litellm` _(optional)_ | LLM gateway used by CrewAI plugin |
| `globus-compute-endpoint` _(optional)_ | Distributed HPC execution |

---

## Configuration

All settings live in a single YAML file. A fully-annotated reference is at [configs/simulator_config.yaml](configs/simulator_config.yaml).

The file has six top-level sections: `simulator`, `poc`, `flowcept`, `report`, `models`/`temperatures`, and `paths`.

### Simulator core

```yaml
simulator:
  topics:
    - "Your research topic here"

  stagger_delay: 5.0            # Seconds between agent startups (thundering-herd guard)
  status_interval_seconds: 60.0 # Print leaderboard snapshot every N seconds (0 = off)
  max_duration_seconds: null    # Auto-stop after N seconds (null = run until PoC finishes)
```

### Research topics and PDF topics

Topics can be plain text strings **or paths to PDF files**. When a PDF path is given the simulator extracts the document text (up to 16,000 characters) and uses it as the research context for every agent — generation prompts, review prompts, and RAG queries all receive the document content instead of a short string.

```yaml
simulator:
  topics:
    # Plain text topic
    - "Explore the intersection of vector databases and knowledge graphs for HPC memory"

    # PDF topic — absolute or relative path to an existing PDF
    - "/path/to/paper.pdf"
    - "research_papers/2021_Highly_accurate_protein_structure_prediction_with_AlphaFold.pdf"
```

CLI override (single topic):

```bash
python -m academy_coscientist.simulator \
    --config configs/simulator_config.yaml \
    --topic "/path/to/my_paper.pdf"
```

Multiple topics cycle round-robin across generation agents; you can freely mix PDF paths and plain text strings in the same list.

> **Requires**: `pypdf` is installed automatically with the package. For very large PDFs only the first 16,000 characters are used.

### Team mode

Comment out or remove the `teams:` block to fall back to flat agent mode.

```yaml
simulator:
  teams:
    count: 2                   # Number of chained teams (see chaining rules below)
    reviewers_per_team: 2      # ReviewAgents per team
    hypotheses_per_cycle: 3    # Hypotheses per team loop cycle
    score_threshold: 0.75      # Min avg score to pass Team 0 (each team +0.10, capped 0.95)
    min_pass_score: 0.50       # Below this after max rounds → discard
    max_refinement_rounds: 5   # Max refine-then-re-review iterations per hypothesis
    interval: 60.0             # Seconds between team cycles
    max_cycles: 2              # Teams stop after N cycles (null = unlimited)
```

**How N teams chain together:**

- **Team 0 only** runs a `HypothesisGenerationAgent`; all other teams are pure refiners.
- Validated hypotheses from Team i flow into `staging_tournament[i]`, which Team i+1 reads from.
- **Every team** also pushes directly to the global leaderboard — not just the final team.
- Pass threshold escalates: Team 0 = `score_threshold`, Team 1 = `+0.10`, Team 2 = `+0.20`, … capped at 0.95.

Example with `count: 3` and `score_threshold: 0.75`:

```
Team 0  generate + refine  threshold 0.75  → staging[0] + global
Team 1  refine staging[0]  threshold 0.85  → staging[1] + global
Team 2  refine staging[1]  threshold 0.95  → global only
```

### Flat mode (no teams)

```yaml
simulator:
  agents:
    generation: 2
    review:     3
    tournament: 1
    meta:       1
    vectordb:   1
    harvester:  1

  intervals:
    generation:  60   # seconds
    review:      45
    tournament:  30
    meta:        120

  hypotheses_per_cycle: 3
```

### RAG and paper harvesting

The harvester tries paper sources in order until one returns results:

| Priority | Source | Key required | Notes |
|---|---|---|---|
| 1 | **Semantic Scholar** | optional (`SEMANTIC_SCHOLAR_API_KEY`) | most-cited papers, best coverage |
| 2 | **OpenAlex** | none (`OPENALEX_EMAIL` for higher limits) | fully open, good citation data |
| 3 | **arXiv** | none | newest preprints; no citation counts |

```yaml
simulator:
  rag_top_k: 5           # Abstracts injected into generation/review prompts (0 = off)

  harvester:
    papers_per_topic: 20
    min_citation_count: 10   # 0 = no filter (always 0 for arXiv results)
    download_pdfs: true
    loop_interval: 604800    # Seconds between harvest runs (604800 = weekly)
    loop_start_delay: 0      # 0 = harvest immediately on startup

paths:
  docs_dir:            "research_papers"
  embeddings_dir:      "embeddings"
  abstracts_cache_dir: "embeddings/abstracts"
```

### Exchange backends

```yaml
simulator:
  exchange:
    type: local          # local | redis | http
    redis_host: localhost
    redis_port: 6379
    address: "https://exchange.academy-agents.org"  # for type: http
    auth_method: "globus"

  executor:
    type: thread         # thread | process
    max_workers: 32
```

### Proof-of-concept agent

```yaml
poc:
  enabled: true          # Set to false to skip PoC execution entirely
  top_n: 3               # Number of top-ranked hypotheses to turn into experiments
  max_retries: 3         # Fix-and-retry attempts if the generated script fails
  output_dir: null       # null → auto: <run_dir>/poc_results

  # Docker execution sandbox
  docker_image:   "academy-poc-runner"  # Build first: docker build -t academy-poc-runner docker/
  code_timeout:   90                    # Wall-clock limit per execution (seconds)
  docker_memory:  "512m"               # Container memory cap
  docker_network: "none"               # "none" = air-gapped; "bridge" if internet needed
```

Code is passed to the container via **stdin** (`python -`) — no host filesystem mounts.
If the Docker daemon is unreachable, the PoC agent logs a warning and the execution step returns a non-zero code; interpretation still runs on the empty output.

**Build the PoC sandbox image** (one-time setup):

```bash
docker build -t academy-poc-runner docker/
```

The image ships numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, statsmodels, sympy, xgboost, lightgbm, pyarrow, h5py, networkx, scikit-image, Pillow, tqdm, joblib, requests, and tabulate — enough for most numeric experiments without internet access inside the container. Use `docker_network: "bridge"` if the generated code needs to download additional packages.

### FlowCept provenance tracking

```yaml
flowcept:
  enabled: true
  workflow_name: "academy-coscientist"
  dump_path: null          # null → auto: <run_dir>/provenance_buffer.jsonl
  perf_csv: null           # null → auto: <run_dir>/provenance_perf.csv
  performance_tracking: true
```

Query provenance after a run:
```bash
python -c "from flowcept import Flowcept; [print(r) for r in Flowcept.read_buffer_file()]"
```

### PDF report

```yaml
report:
  enabled: true
  output_filename: "report_{date}.pdf"  # {date} → YYYYMMDD_HHMMSS
  top_n: 5000                           # Max hypotheses included in full detail
  output_dir: null                      # null → auto: <run_dir>
```

### Argo gateway (ANL only)

When `argo_mode` is enabled **all** LLM calls are routed through the Argo gateway instead of OpenAI/Anthropic directly. No `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is needed in this mode.

```yaml
argo_mode: true        # Route all LLM calls through Argo (requires ANL username)
argo_user: "your_anl_username"   # Or set the ARGO_USER environment variable
```

```bash
export ARGO_USER="your_anl_username"
python -m academy_coscientist.simulator --config configs/simulator_config.yaml
```

Argo model names differ from OpenAI/Anthropic names — use the Argo-specific identifiers in the `models:` section (e.g. `gpt4o`, `gpt54`, `claudeopus46`, `gemini25pro`).

### LLM models and temperatures

Each agent type has its own model and temperature knob so you can tune brainstorming creativity independently from PoC code generation precision:

```yaml
models:
  # --- Per-agent-type models ---
  generation:    "gpt4o"                 # HypothesisGenerationAgent — brainstorming
  review:        "gpt4o"                 # ReviewAgent — scoring and critique
  refinement:    "gpt54"                 # HypothesisRefinerAgent — rewriting
  poc_codegen:   "claudeopus46"          # PoC code generation
  poc_fix:       "claudeopus46"          # PoC code fixing (auto-retry)
  poc_interpret: "claudeopus46"          # PoC result interpretation
  report:        "gemini25pro"           # PDF report narrative sections
  commentary:    "gpt52"                 # Agent self-commentary (all agents)
  # --- Infrastructure ---
  embedding:     "local-all-MiniLM-L6-v2"  # Local FAISS (no API key needed)
  argo:          "gpt54"                 # Default when no role-specific model matches

temperatures:
  generation:    0.9   # HypothesisGenerationAgent — brainstorming new hypotheses
  review:        0.5   # ReviewAgent — deterministic scoring and critique
  refinement:    0.5   # HypothesisRefinerAgent — focused improvement
  poc_codegen:   0.4   # PoC code generation — creative but focused
  poc_fix:       0.3   # PoC code fixing — deterministic bug fixing
  poc_interpret: 0.2   # PoC result interpretation — factual, precise
  report:        0.2   # PDF report narratives
  commentary:    0.5   # Agent self-commentary
```

Without `argo_mode`, model names should be OpenAI or Anthropic model IDs (e.g. `gpt-4o`, `claude-opus-4-6`).

---

## Usage

### Autonomous simulator

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

python -m academy_coscientist.simulator --config configs/simulator_config.yaml
```

CLI overrides:

```bash
# Override topic (plain text)
python -m academy_coscientist.simulator \
    --config configs/simulator_config.yaml \
    --topic "Quantum error correction"

# Override topic with a PDF file — agents use the document content as research context
python -m academy_coscientist.simulator \
    --config configs/simulator_config.yaml \
    --topic "/path/to/paper.pdf"

# Limit runtime
python -m academy_coscientist.simulator \
    --config configs/simulator_config.yaml \
    --max-duration 3600

# More hypotheses per cycle
python -m academy_coscientist.simulator \
    --config configs/simulator_config.yaml \
    --hypotheses-per-cycle 5
```

The simulator runs until the PoC agent finishes testing the top hypotheses, then stops automatically. Press `Ctrl-C` to stop early at any time.

---

## Output

All outputs for a single run are written to one directory: `runs/<run_id>/`.

```
runs/
└── 20260309_143012_abc123/       ← one directory per run
    ├── events.log                # human-readable event log
    ├── actions.jsonl             # all log_action() records (reviews, refinements, …)
    ├── llm_calls.jsonl           # full LLM call records with token counts
    ├── provenance_buffer.jsonl   # FlowCept provenance dump
    ├── provenance_perf.csv       # per-call wall-clock overhead
    ├── report_20260309_143512.pdf  # comprehensive PDF report
    └── poc_results/
        ├── abc123_rank1/
        │   ├── poc_code.py       # generated (and possibly auto-fixed) Python script
        │   └── report.json       # verdict, confidence, interpretation, next steps
        └── def456_rank2/
            └── ...
```

Paths default to `<run_dir>/…` but can be overridden in the config (see `poc.output_dir`, `report.output_dir`, `flowcept.dump_path`).

### Leaderboard (live)

Printed to stdout every `status_interval_seconds`:

```
============================================================
  LEADERBOARD  (3 hypotheses)
============================================================
  # 1  score=0.708  Improved Semantic Retrieval Accuracy via Graph-Augmented Embedding
  # 2  score=0.675  Reduced Query Latency via Hybrid Vector-Graph Indexing
  # 3  score=0.613  Memory Footprint Reduction via Semantic Partitioning
============================================================
```

### Proof-of-concept results

`poc_results/<hypothesis_id>_rank<N>/report.json` fields:

| Field | Description |
|---|---|
| `verdict` | `SUPPORTS` / `REFUTES` / `INCONCLUSIVE` |
| `confidence` | 0.0 – 1.0 |
| `interpretation` | Claude's plain-language explanation of results |
| `next_steps` | Suggested follow-up experiments |
| `returncode` | Subprocess exit code (0 = success) |

### PDF report

Generated at run end. Includes:
- Executive summary (Claude-written narrative)
- Hypothesis leaderboard with per-reviewer dimension scores
- Full refinement history per hypothesis (all review rounds)
- PoC experiment code and results
- LLM and agent activity statistics
- Provenance performance timing

### Provenance

`actions.jsonl` and `llm_calls.jsonl` contain the raw structured records used to build the report. `provenance_buffer.jsonl` is the FlowCept dump with one JSON record per agent action and LLM call.

---

## Multi-Framework Provenance

Academy Co-Scientist ships FlowCept plugins for four agentic frameworks. Each plugin captures a full provenance hierarchy and all plugins can share a single FlowCept buffer so every record in a mixed pipeline carries the same `campaign_id` and `workflow_id`.

### Plugins

| Plugin | File | Framework |
|---|---|---|
| `FlowceptAcademyPlugin` | `plugins/flowcept_plugin.py` | Academy |
| `FlowceptLangGraphPlugin` | `plugins/flowcept_langgraph_plugin.py` | LangGraph |
| `FlowceptCrewAIPlugin` | `plugins/flowcept_crewai_plugin.py` | CrewAI |
| `FlowceptAutoGenPlugin` | `plugins/flowcept_autogen_plugin.py` | AutoGen |

### Provenance hierarchy

```
WorkflowObject  (one per pipeline run)
  └─ TaskObject  subtype=crewai_crew    activity_id="my-crew"
       └─ TaskObject  subtype=crewai_task    activity_id="research_papers"
            └─ TaskObject  subtype=crewai_agent  activity_id="Researcher"
                 └─ TaskObject  subtype=tool_call   activity_id="search_papers"
                 └─ TaskObject  subtype=llm_call    activity_id="gpt-4o-mini"
  └─ TaskObject  subtype=autogen_run   activity_id="research-team"
       └─ TaskObject  subtype=autogen_message  activity_id="assistant"
       └─ TaskObject  subtype=autogen_message  activity_id="critic"
```

### Sharing a single buffer across frameworks

```python
from academy_coscientist.plugins.flowcept_plugin import FlowceptAcademyPlugin
from academy_coscientist.plugins.flowcept_langgraph_plugin import FlowceptLangGraphPlugin
from academy_coscientist.plugins.flowcept_crewai_plugin import FlowceptCrewAIPlugin
from academy_coscientist.plugins.flowcept_autogen_plugin import FlowceptAutoGenPlugin

academy_plugin = FlowceptAcademyPlugin(config={"enabled": True, "dump_path": "provenance.jsonl"})
academy_plugin.start()

lg_plugin    = FlowceptLangGraphPlugin.from_academy_plugin(academy_plugin)
crewai_plugin  = FlowceptCrewAIPlugin.from_academy_plugin(academy_plugin)
autogen_plugin = FlowceptAutoGenPlugin.from_academy_plugin(academy_plugin)

# … run your pipeline …

academy_plugin.stop()
# All four frameworks → one provenance.jsonl, one campaign_id, one workflow_id
```

### CrewAI hook-based capture

The CrewAI plugin uses two extension points:

1. **Event bus** (`BaseEventListener`) — crew kickoff, task, and agent lifecycle events
2. **LLM hooks** (`register_before/after_llm_call_hook`) — full message lists, response text, agent role/goal/backstory, task description, iteration count
3. **Tool hooks** (`register_before/after_tool_call_hook`) — typed tool input dict, tool result, agent/task/crew references

This gives richer provenance than the event bus alone, with one `llm_call` record per LLM invocation containing both `used` (messages sent) and `generated` (response received).

### Examples

| Script | What it demonstrates |
|---|---|
| `examples/crewai_provenance_example.py` | 2-agent CrewAI crew with stub LLM; shows crew/task/agent/llm/tool records |
| `examples/autogen_provenance_example.py` | 2-agent AutoGen `RoundRobinGroupChat` with stub client; shows run + message records |
| `examples/combined_four_framework_provenance.py` | Full 4-framework pipeline (Academy → LangGraph → CrewAI → AutoGen) sharing one buffer |

Run any example (no API key required):

```bash
source venv/bin/activate
python examples/crewai_provenance_example.py
python examples/autogen_provenance_example.py
python examples/combined_four_framework_provenance.py
```

---

## Project Structure

```
academy_coscientist/
├── agents/
│   ├── generation_agent.py       # HypothesisGenerationAgent
│   ├── review_agent.py           # ReviewAgent
│   ├── tournament_agent.py       # TournamentAgent (ranking + dedup)
│   ├── refiner_agent.py          # HypothesisRefinerAgent
│   ├── team_coordinator_agent.py # TeamCoordinatorAgent
│   ├── meta_agent.py             # MetaReviewAgent (literature-grounded)
│   ├── poc_agent.py              # ProofOfConceptAgent (orchestrates generate → execute → interpret)
│   ├── docker_executor_agent.py  # DockerExecutorAgent (isolated code sandbox)
│   ├── paper_harvester_agent.py  # PaperHarvesterAgent (S2 → OpenAlex → arXiv)
│   ├── research_vector_agent.py  # ResearchVectorDBAgent (FAISS)
│   └── provenance_report_agent.py # ProvenancePDFReportAgent
│
├── utils/
│   ├── config.py                 # YAML config loader
│   ├── utils_llm.py              # LLM call wrappers (brainstorm, review, refine, PoC)
│   ├── utils_logging.py          # Structured logging + plugin hook system
│   └── utils_papers.py           # Shared paper search (S2 → OpenAlex → arXiv)
│
├── plugins/
│   ├── flowcept_plugin.py              # FlowCept provenance plugin (Academy)
│   ├── flowcept_langgraph_plugin.py    # FlowCept provenance plugin (LangGraph)
│   ├── flowcept_crewai_plugin.py       # FlowCept provenance plugin (CrewAI — event bus + LLM/tool hooks)
│   └── flowcept_autogen_plugin.py      # FlowCept provenance plugin (AutoGen)
│
└── simulator.py                  # Autonomous multi-agent simulator (entry point)

configs/
└── simulator_config.yaml         # Fully-annotated simulator configuration

examples/
├── crewai_provenance_example.py            # CrewAI 2-agent crew, stub LLM, provenance demo
├── autogen_provenance_example.py           # AutoGen RoundRobinGroupChat, stub client, provenance demo
└── combined_four_framework_provenance.py   # Academy + LangGraph + CrewAI + AutoGen shared buffer

docs/
└── ARCHITECTURE.md               # Detailed architecture and flow documentation

runs/                             # One sub-directory per run (auto-created)
└── <run_id>/
    ├── events.log
    ├── actions.jsonl
    ├── llm_calls.jsonl
    ├── provenance_buffer.jsonl
    ├── provenance_perf.csv
    ├── report_<date>.pdf
    └── poc_results/

research_papers/                  # Harvested PDFs and abstracts (auto-created)
embeddings/                       # FAISS index and abstract cache (auto-created)
```

---

## Distributed Execution

The exchange backend determines how agents communicate. For distributed execution across HPC nodes or cloud resources, use the Globus/HTTP backend:

```yaml
simulator:
  exchange:
    type: http
    address: "https://exchange.academy-agents.org"
    auth_method: "globus"
```

Set up a Globus Compute endpoint on each remote resource:

```bash
globus-compute-endpoint configure co-scientist-endpoint
globus-compute-endpoint start co-scientist-endpoint
```

For multi-process local execution (true parallelism within one node):

```yaml
simulator:
  executor:
    type: process
    max_workers: 8
```

> **Note**: Process mode requires all agent classes and their dependencies to be picklable. Thread mode (default) is safer for local development.
