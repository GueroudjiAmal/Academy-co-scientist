
# Academy Co-Scientist Tutorial

Welcome to the **Academy Co-Scientist** tutorial repository. This project showcases how to build and orchestrate a multi-agent research assistant pipeline using [Academy](https://github.com/academy-agents/academy). Agents collaborate to generate hypotheses, review ideas, and synthesize reports — all autonomously.

---

## Table of Contents

- [Features](#features)
- [Setup](#setup)
  - [Install](#install)
  - [Globus Compute (Optional)](#globus-compute-optional)
- [Configuration](#configuration)
- [Usage](#usage)
- [Agents Overview](#agents-overview)
- [Project Structure](#project-structure)

---

## Features

- Modular multi-agent research assistant
- Hypothesis generation, peer-review, tournament selection, and meta-review
- Vector-based document embedding and retrieval
- Compatible with `academy`'s `Manager` for distributed execution
- Fallback to direct mode if `Manager` not found
- Reproducible pipeline with config-based customization

---

## Setup

### Install

We recommend using a virtual environment.

```bash
git clone https://github.com/YOUR_USERNAME/academy-coscientist.git
cd academy-coscientist
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Globus Compute (Optional)

If you're running across distributed resources (e.g., HPC), you can configure a Globus Compute endpoint:

```bash
globus-compute-endpoint configure co-scientist-endpoint
globus-compute-endpoint start co-scientist-endpoint
```

---

## Configuration

Edit the `config.yml` file in the root directory:

```yaml
launch:
  topic: "AI Reliability enhancement"
  embeddings_dir: "embeddings"
  docs_dir: "DOCs"
  abstracts_cache_dir: "embeddings/abstracts"
```

> You can override the topic at runtime using `--topic` CLI argument.

---

## Usage

Run the pipeline using:

```bash
python -m academy_coscientist.launcher --config config.yml
```

Or override the topic:

```bash
python -m academy_coscientist.launcher --config config.yml --topic "Your new topic"
```

The final output will be a report with top-ranked hypotheses and analysis.

---

## Agents Overview
|-------------------------------------------------------------------------------------|
| Agent                 | Role                                                        |
|-----------------------|-------------------------------------------------------------|
| `GenerationAgent`     | Generates initial ideas from the topic                      |
| `ReviewAgent`         | Evaluates and scores hypotheses                             |
| `TournamentAgent`     | Runs pairwise comparisons to rank hypotheses                |
| `MetaAgent`           | Analyzes reviewer consistency and refines rankings          |
| `ReportAgent`         | Synthesizes the final report                                |
| `SupervisorAgent`     | Orchestrates all other agents                               |
| `VectorStoreAgent`    | Handles embedding and document retrieval                    |
|-------------------------------------------------------------------------------------|
---

## Project Structure

```
academy_coscientist/
│
├── agents/                # All agent definitions
│   ├── generation_agent.py
│   ├── review_agent.py
│   └── ...
│
├── utils/                 # Utility functions
│   ├── config.py
│   ├── vector_store.py
│   └── ...
│
├── launcher.py           # Main pipeline launcher
├── config.yml            # YAML config file
└── README.md             # This file
```
