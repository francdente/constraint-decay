# Constraint Decays

Reproducibility package for **"Constraint Decay: The Fragility of LLM Agents in Backend Code Generation"**.

This repository evaluates how LLM-based coding agents (Mini-SWE-Agent, OpenHands) handle structural constraints, architectural patterns, database engines, and ORMs, in multi-file REST API generation. By fixing a unified API contract (the RealWorld Conduit API) across 80 generation tasks and 20 feature-implementation tasks spanning eight web frameworks, it isolates the effect of constraint accumulation on agent performance.

## Results Reproducibility

All the results from the paper are reproducible using the repository scripts.

Additionally, a zip archive containing the experiments results is available on [HuggingFace](https://huggingface.co/datasets/constraint/constraint_decay/tree/main): [https://huggingface.co/datasets/constraint/constraint_decay/tree/main](https://huggingface.co/datasets/constraint/constraint_decay/tree/main)

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
uv sync
cp .env.template .env
# fill in LLM_API_KEY, LLM_MODEL, paths
# unzip results for full trajectories of experiments ran in the paper and place them in data/results to reproduce all paper tables
```

`uv sync` creates a new virtual environment in the current folder automatically.

## Reproducing Paper Tables

All paper tables can be reproduced directly from the raw `data/results/` directory after downloading it.

### Tables 1, 1-raw, 2a, 2b, 3 (main results)

```bash
uv run evaluation_tables.py data/results           # terminal output
uv run evaluation_tables.py data/results --latex    # also emit LaTeX
```

This produces:
- **Table 1**: A% and pass@1 across constraint levels L0–L3, with ΔA% (L3−L0)
- **Table 1-raw**: Same as Table 1 but without verifier enforcement (Appendix E)
- **Table 2a**: Marginal effect of each constraint via matched-pair differences (±SEM)
- **Table 2b**: Feature-implementation pass@1 per model and agent
- **Table 3**: Per-framework A% ranking with pass@1 subscripts — the framework
  leaderboard reported in the RQ2 appendix (8 frameworks × 8 model–scaffold
  configurations, including MiniMax-M2.5 and GPT-5.4)

A% is computed **per task, then averaged across tasks**. Pooling runs directly
gives a different answer wherever a task has fewer than 3 completed runs; all
paper numbers use the per-task convention.

### RQ2 framework sensitivity figure

```bash
uv run plot_framework_sensitivity.py data/results \
  --output figures/fig_framework_sensitivity.pdf
```

Produces the RQ2 figure: A% per framework across six model–scaffold
configurations, aggregated over constraint levels, with ±1 standard error bands
computed across individual runs. The plotted subset is the top-3 and bottom-3
frameworks by average A% (Express, Koa, Flask / Django, FastAPI, Hono);
aiohttp and Fastify rank fourth and fifth and are omitted for legibility.
Configurations and frameworks are set by the `CONFIGS` and `FRAMEWORKS`
constants at the top of the script; `N_MID_TIER` positions the mid-tier/strong
divider.

Two configurations present in `data/results/` are deliberately excluded from
this figure: `openhands_sdk/minimax-m2.5` (covers only 4 frameworks, not a full
sweep) and `qwen3-235b` (near-zero scores leave all frameworks within noise).
Both remain in the Table 3 appendix leaderboard.

### Subset representativeness (Appendix, `tab:more_results`)

```bash
uv run evaluation_subset_representativeness.py data/results           # terminal output
uv run evaluation_subset_representativeness.py data/results --latex   # also emit LaTeX
```

Compares Assert% per constraint level (L0–L3) on the full task set vs. the
subset used for cost-constrained models (aiohttp, Express, FastAPI, Fastify),
for every configuration evaluated on the full benchmark. Reports per-level
Δ (full − subset), per-level MAE, and the Pearson/Spearman correlations over
all paired (full A%, subset A%) observations. Full-evaluation configurations
are detected automatically (≥ 90% coverage of the benchmark task set).

### Verifier impact (Appendix, `tab:verifier_comparison`)

```bash
uv run evaluation_verifier_impact.py data/results           # terminal output
uv run evaluation_verifier_impact.py data/results --latex   # also emit LaTeX
```

Compares, per configuration and constraint level (L0–L3), A% with verifier
enforcement (runs violating any static verifier have their assertion score
zeroed) against Raw A% without enforcement, with Δ = Raw A% − A%. Also prints
the summary statistics cited in the appendix text: maximum Δ across all
configurations and levels, average L0→L3 decay under both variants, and the
share of constrained runs violating structural constraints (overall and among
test-passing runs).

### Table 4 (failure taxonomy)

The RQ3 taxonomy covers four Mini-SWE-Agent configurations, all on the full task
set: Qwen3-Coder-Next, GPT-5-mini, MiniMax-M2.5 and GPT-5.4.

The judge classifications are pre-computed in `data/failure_analysis_*.csv` and
`data/logic_subcategories*.csv`. To build the paper table from them:

```bash
uv run failure_analysis_aggregate.py --data-dir data           # terminal output
uv run failure_analysis_aggregate.py --data-dir data --latex   # also emit LaTeX
```

This produces the coarse category distribution (% of failed runs), the
logic-error subcategory distribution (% of logic errors), and the data-layer
defect summary (incorrect query logic + DB/ORM runtime errors) per model.

To re-run the LLM judge from scratch (one invocation per model):

```bash
# Coarse failure categories
uv run failure_analysis.py data/results --agent mini_swe_sdk --model qwen3-coder-next \
  --output data/failure_analysis_qwen3-coder-next.csv
uv run failure_analysis.py data/results --agent mini_swe_sdk --model gpt-5-mini \
  --output data/failure_analysis_gpt-5-mini.csv
uv run failure_analysis.py data/results --agent mini_swe_sdk --model minimax-m2.5 \
  --output data/failure_analysis_minimax-m2.5.csv
uv run failure_analysis.py data/results --agent mini_swe_sdk --model gpt-5.4 \
  --output data/failure_analysis_gpt-5.4.csv

# Logic-error subcategories
uv run failure_analysis_logic.py data/results --agent mini_swe_sdk --model qwen3-coder-next \
  --output data/logic_subcategories.csv
uv run failure_analysis_logic.py data/results --agent mini_swe_sdk --model gpt-5-mini \
  --output data/logic_subcategories_gpt-5-mini.csv
uv run failure_analysis_logic.py data/results --agent mini_swe_sdk --model minimax-m2.5 \
  --output data/logic_subcategories_minimax-m2.5.csv
uv run failure_analysis_logic.py data/results --agent mini_swe_sdk --model gpt-5.4 \
  --output data/logic_subcategories_gpt-5.4.csv
```

### Token consumption tables (Appendix, `tab:tokens_global` and `tab:tokens_per_pair`)

```bash
uv run evaluation_token_consumption.py data/results           # terminal output
uv run evaluation_token_consumption.py data/results --latex   # also emit LaTeX
```

Produces the two Appendix token-consumption tables: global input/output token
consumption by constraint level (L0–L3) aggregated across all agent–model
configurations, and average input/output tokens per run by level and
agent–model pair (K = ×10³). Subset evaluations are detected automatically
(≤ 50% coverage of the largest task set) and starred. Token extraction reuses
`tokens.py`: OpenHands usage is read from `base_state.json`; Mini-SWE usage is
inferred by simulating cumulative context with tiktoken.

### Detailed per-run statistics

```bash
# All agents and models
uv run evaluation.py data/results --summary

# Specific agent/model
uv run evaluation.py data/results --agent mini_swe_sdk --model gpt-5-mini --summary

# Feature-implementation tasks
uv run evaluation_feature.py data/results
```


## Running New Experiments

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key for the LLM provider |
| `LLM_MODEL` | Model identifier (e.g., `openai/gpt-5-mini`) (*)|
| `LLM_BASE_URL` | Optional custom endpoint (e.g., for vLLM self-hosted models) |
| `TASKS_ABSOLUTE_PATH` | Absolute path to `data/tasks/` |
| `RESULTS_ABSOLUTE_PATH` | Absolute path to `data/results/` |

---

(*) When using an external provider, model name should follow LiteLLM conventions (see https://docs.litellm.ai/docs/providers).

### Run agent on generation tasks

```bash
# Single task
uv run main.py --agent mini_swe_sdk --task uv/uv-flask-openapi-unconstrained.json

# All tasks for node runtime, 3 runs each
uv run main.py --agent mini_swe_sdk --task node --runs 3

# Evaluate only (skip agent run, use existing patches in data/results/)
uv run main.py --agent mini_swe_sdk --task node --evaluate --runs 3
```

### Run agent on feature-implementation tasks

```bash
# Single task
uv run main.py --agent mini_swe_sdk --task express/hard-focused-task-001.json

# All tasks for express, 3 runs each
uv run main.py --agent mini_swe_sdk --task express --runs 3

# Evaluate only (skip agent run, use existing patches in data/results/)
uv run main.py --agent mini_swe_sdk --task express --runs 3 --evaluate
```

## Repository Structure

### Scripts

| Script | Purpose | Paper Reference |
|--------|---------|-----------------|
| `main.py` | End-to-end pipeline: run an agent on tasks and evaluate results | §3, §4 |
| `run_agent.py` | Execute an LLM agent (Mini-SWE, OpenHands) inside a Docker container | §4 (Agents) |
| `evaluate.py` | Apply an agent's patch in a clean container, run Newman behavioral tests, collect results | §3.3 (Evaluation Pipeline) |
| `evaluation.py` | Aggregate per-run Newman CSVs into task-level and global statistics (A%, pass@k), run static verifiers | §4 (Metrics) |
| `evaluation_feature.py` | Same aggregation for feature-implementation tasks | Table 2b |
| `evaluation_tables.py` | **Reproduce all paper tables** from raw results (see below) | Tables 1–3, 2a, 2b |
| `evaluation_subset_representativeness.py` | Full-set vs. subset A% comparison with Pearson/Spearman correlations | Appendix, subset validity |
| `evaluation_verifier_impact.py` | A% with vs. without verifier enforcement per constraint level | Appendix, verifier impact |
| `failure_analysis.py` | Classify failed runs into coarse failure categories using an LLM judge | §5.3 (RQ3), Table 4 |
| `failure_analysis_logic.py` | Sub-classify logic errors into root-cause categories | §5.3, Table 4 (bottom) |
| `failure_analysis_aggregate.py` | **Aggregate the per-model judge CSVs into the RQ3 taxonomy table** across all four models | §5.3, Table 4 |
| `plot_framework_sensitivity.py` | **Render the RQ2 framework-sensitivity figure** from raw results | §5.2 (RQ2), Figure |
| `evaluation_token_consumption.py` | **Reproduce the token-consumption tables** by constraint level and agent–model pair | Appendix (Token Consumption) |
| `tokens.py` | Count input/output tokens from agent trajectories (extraction backend for the above) | Appendix (Token Consumption) |
| `generate_task.py` | Generate task JSON files from repository metadata | §3.1 |
| `generation_tasks.py` | Batch-generate all 80 generation task definitions | §3.1 |

### Core Library (`constraint_decay/`)

| Module | Purpose |
|--------|---------|
| `model.py` | Data models: `Task`, `RepoMetadata`, `TaskMetadata` |
| `dockerized_cmd.py` | Docker Compose command wrappers (build, run, exec, copy, logs) |
| `verifiers.py` | Static constraint verifiers: clean architecture (4-layer + dependency direction), database type, ORM compliance |
| `prompts/` | Prompt templates for agents and LLM judges |

### Data Artifacts (`data/`)

| Path | Description | Paper Reference |
|------|-------------|-----------------|
| `tasks/uv/` | 40 generation task JSONs for Python frameworks (Flask, FastAPI, Django, aiohttp) | §3.1 |
| `tasks/node/` | 40 generation task JSONs for Node.js frameworks (Express, Fastify, Hono, Koa) | §3.1 |
| `tasks/{aiohttp,express,fastapi,flask,honojs}/` | 20 feature-implementation task JSONs (4 per framework) | Appendix D |
| `results/` | Raw evaluation results: agent patches, Newman CSVs, server logs, trajectories | All tables |
| `failure_analysis_qwen3-coder-next.csv` | Coarse failure classifications, Qwen3-Coder-Next (194 failed runs) | Table 4 (top) |
| `failure_analysis_gpt-5-mini.csv` | Coarse failure classifications, GPT-5-mini (200 failed runs) | Table 4 (top) |
| `failure_analysis_minimax-m2.5.csv` | Coarse failure classifications, MiniMax-M2.5 (145 failed runs, full task set) | Table 4 (top) |
| `failure_analysis_gpt-5.4.csv` | Coarse failure classifications, GPT-5.4 (103 failed runs) | Table 4 (top) |
| `logic_subcategories.csv` | Logic-error root causes — **Qwen3-Coder-Next only** (137). Also holds a superseded 16-task-subset MiniMax-M2.5 analysis (56 rows). | Table 4 (bottom) |
| `logic_subcategories_gpt-5-mini.csv` | Logic-error root causes, GPT-5-mini (63) | Table 4 (bottom) |
| `logic_subcategories_minimax-m2.5.csv` | Logic-error root causes, MiniMax-M2.5 (111, full task set) | Table 4 (bottom) |
| `logic_subcategories_gpt-5.4.csv` | Logic-error root causes, GPT-5.4 (64) | Table 4 (bottom) |
| `judge_validation_set.json` | 50-sample stratified validation set with human + judge labels | Appendix B |
| `judge_validation_metrics.json` | Per-category precision/recall/F1 and Cohen's κ | Appendix B, Table 6 |

### Results Directory Layout

```
data/results/{runtime}/{agent}/{model}/{task}/{date}/run_N/
├── {agent}-{task}.patch       # Agent-generated unified diff
├── newman-run-report-*.csv    # Per-assertion pass/fail results
├── conduit.log                # Server stdout/stderr
├── setup.log                  # Dependency install output
├── {trajectory}               # Agent trajectory in the given run
└── newman-run-report-*.html   # Newman HTML report
```

Generation tasks use `runtime` = `uv` (Python) or `node` (Node.js). Feature-implementation tasks use framework-level directories (`express`, `flask`, `fastapi`, `aiohttp`, `honojs`).

### Runtime Environment (`runtime/`)

| Path | Description |
|------|-------------|
| `agents/` | Agent implementations: `mini_swe_sdk.py`, `openhands_sdk.py`, `mock_agent.py`, YAML configs |
| `{aiohttp,express,fastapi,flask,honojs,...}/` | Per-framework Docker Compose projects (Dockerfile, docker-compose.yml) |
| `newman/` | Newman test runner container |
| `openapi.yml` | Full OpenAPI 3.0 specification (19 endpoints, ~500 lines) |

### Task naming convention

Generation task filenames encode constraints:

```
{framework}-{spec_type}-{constraints}.json

# Examples:
flask-openapi-unconstrained              # L0: no constraints
express-openapi-clean_architecture       # L1: architecture only
aiohttp-openapi-clean_architecture-postgres  # L2: architecture + DB
hono-openapi-clean_architecture-sqlite-sequelize  # L3: all three
```

## Task generation

All tasks required to reproduce the results are already available in the `data/tasks` folder.

However, it is possible to generate more tasks following the methodology described in the paper simply by modifing the `generation_tasks.py` script.

It is sufficient to add the new task definitions (e.g. runtime / framework) to the top module constants in order to generate all L0 to L3 variations for a new task.

**Example 1: add the Hapi JS framework**

1. Add the `hapi` string to the `node` array in the `FRAMEWORKS` constant;
2. run `uv run generation_tasks.py`.

It is also possible to add tasks for new runtimes, but it requires more steps. Here is a simplified overview of the required steps to add a new runtime.

**Example 2: add support for Go and Gin**

1. Create a Docker runtime for Go (refer to [runtim/node](runtime/node/) as a template);
2. [constraint_decay/prompts](constraint_decay/prompts/): add the prompt templates for Go and the relative ORM (`evaluation`, `mandatory_files` and `requirements`);
3. [generation_tasks.py](generation_tasks.py): add the go references in the top module constants;
4. run `uv run generation_tasks.py`.
