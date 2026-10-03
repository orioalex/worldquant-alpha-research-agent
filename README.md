# WorldQuant Alpha Research Agent

Planner-driven, tool-using alpha research system for WorldQuant BRAIN with:

- stage-aware candidate exploration, exploitation, robustness testing, and harvesting
- controlled simulation/check execution via existing API client
- governance-first submission modes (`disabled`, `manual`, `auto_approved`)
- hypothesis logging and failure-pattern-aware mutation logic
- correlation-aware submission readiness and decorrelation repair candidates
- reproducible JSON reports and baseline-vs-agent evaluation harness
- presentation-ready Streamlit console with a built-in demo case, live controls, visual analytics, and economic logic

Repository: [https://github.com/zeron-G/worldquant-alpha-research-agent](https://github.com/zeron-G/worldquant-alpha-research-agent)

## Project Structure

- `worldquant_brain_cli.py`  
  Low-level API client and CLI (auth, simulate, check, submit, metadata fetch).
- `alpha_research_pipeline.py`  
  Baseline heuristic search pipeline (seed + refine + score + optional submit).
- `alpha_research_agent.py`  
  New planner-driven agent CLI entrypoint.
- `alpha101_ideas.json`  
  WorldQuant BRAIN-compatible transliteration of all 101 formulaic alpha seeds for broad replication/search runs.
- `alpha_agent/`  
  Agent runtime modules:
  - `config.py`: shared runtime dataclasses
  - `planner.py`: heuristic planner + OpenAI JSON planner
  - `research_logic.py`: quant-style stage logic, novelty scoring, check-aware and robustness candidate builders
  - `engine.py`: orchestrator loop, tool execution, submission gating, run reports
  - `evaluation.py`: baseline-vs-agent case-suite runner
- `streamlit_app.py`  
  Presentation console for demo mode, live agent runs, parameter tuning, visual diagnostics, and economic analysis.
- `docs/eval_cases.json`  
  Starter replay case suite.

## Features

### 1) End-to-end Alpha Agent Loop

The agent repeatedly executes:

1. gather frontier context (family performance, failed-check histogram, stage, hypotheses)
2. choose next action (`evaluate_seed`, `evaluate_refine`, `evaluate_diversify`, `evaluate_robustness`, `submit_best`, `stop`)
3. call simulation/check tools
4. update leaderboard, stage, and research notebook
5. log rationale, hypothesis, risk note, and outcomes

All run events are recorded under `<workdir>/agent_runs/*.json`.

### 2) Quant-Style Stage Policy

The agent follows four research stages:

- `explore`: maximize family and expression diversity under budget constraints
- `exploit`: target dominant failure checks with check-aware refinements
- `robustness`: stress-test top candidates across universe/neutralization/truncation perturbations
- `harvest`: attempt controlled submission only when readiness and governance align

Transitions are data-driven by score quality, submit-readiness, and robustness evidence.

### 3) Prompt Contract (OpenAI Planner)

When `--planner-provider openai` is enabled, the planner receives a structured context and must return strict JSON:

```json
{
  "action": "evaluate_refine",
  "batch_size": 3,
  "rationale": "...",
  "hypothesis": "...",
  "focus_family": "news_attention",
  "risk_note": "...",
  "target_alpha_id": null
}
```

This enforces reproducible, auditable planner decisions instead of free-form text.

### 4) Submission Governance

`submission_mode` controls risk:

- `disabled`: never submit
- `manual`: submit only with explicit approval callback
- `auto_approved`: allow unattended submit action when planner selects it

By default, runs are safe (`disabled`).

Submit readiness requires both quality checks and correlation checks to pass. A candidate that clears Sharpe/fitness/turnover but fails `SELF_CORRELATION` or `PROD_CORRELATION` is treated as a repair target, not as submit-ready. The agent also requires harvest stage before any agent-driven submission, so robustness evidence is collected first.

### 5) Pluggable Planning Backends

- `heuristic`: deterministic planner (no model key needed)
- `openai`: OpenAI-compatible JSON planner via `chat/completions`

If OpenAI planner fails or key is missing, behavior falls back safely to heuristic planning.

### 6) Evaluation Harness

Run baseline pipeline and agent on the same case suite and budgets:

- per-case score and latency deltas
- aggregate win rate and average deltas
- JSON report artifact for appendix/demo

## Requirements

- Python 3.10+
- WorldQuant BRAIN account access
- Optional: OpenAI-compatible API key (only if using `--planner-provider openai`)
- Optional: Streamlit for web UI

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Linux Deployment

The repository includes a small Linux process wrapper, `wqagent`, for the common remote workflow. It keeps credentials and research results on the machine and does not require a system-wide installation.

### 1. Clone and install

```bash
git clone https://github.com/zeron-G/worldquant-alpha-research-agent.git
cd worldquant-alpha-research-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10 or newer is required. GPU acceleration is optional for this repository; the WQ simulation and check requests are sent to the WorldQuant BRAIN API.

### 2. Configure credentials

```bash
cp .env.example .env
chmod 600 .env
${EDITOR:-vi} .env
```

Use either `WQB_EMAIL` plus `WQB_PASSWORD`, or a current `WQB_COOKIE_HEADER`. A cookie header is a live credential: never commit it, paste it into an issue, or place it in a public log.

For the deterministic planner:

```dotenv
ALPHA_AGENT_PLANNER_PROVIDER=heuristic
```

For an OpenAI-compatible planner:

```dotenv
ALPHA_AGENT_PLANNER_PROVIDER=openai
ALPHA_AGENT_PLANNER_MODEL=your-model-id
ALPHA_AGENT_PLANNER_BASE_URL=https://your-provider.example/v1
ALPHA_AGENT_PLANNER_API_KEY_ENV=OPENAI_API_KEY
OPENAI_API_KEY=your-api-key
```

The model provider must support `POST /chat/completions` and JSON response format. The planner is optional; if it is unavailable, the agent falls back to the deterministic heuristic planner.

Keep submission disabled while testing:

```dotenv
ALPHA_AGENT_SUBMISSION_MODE=disabled
```

Only change it to `manual` or `auto_approved` after checking the generated expressions and WQ readiness results. `auto_approved` submits only after the agent reaches harvest and finds a candidate that passes all blocking quality and correlation checks.

### 3. Run the Linux wrapper

```bash
chmod u+x wqagent start_alpha_agent.sh stop_alpha_agent.sh status_alpha_agent.sh

./wqagent s       # start
./wqagent t       # status and latest report summary
./wqagent h       # readable alpha history
./wqagent h 20    # show the top 20 records
./wqagent q       # quality-ready records
./wqagent p       # submit-ready records
./wqagent u       # submission history
./wqagent e       # latest planner events
./wqagent d       # stop
./wqagent r       # restart
```

`./wqagent l` follows the process log and can be stopped with `Ctrl-C`. The start wrapper loads `.env` before expanding the command-line defaults, so `ALPHA_AGENT_SUBMISSION_MODE` and the other runtime settings are honored.

The wrappers use their own directory as the application directory. Set `ALPHA_AGENT_APP_DIR` only when the checkout and runtime files are intentionally separated.

### 4. Inspect generated artifacts

The default work directory is `.alpha_agent`:

```text
.alpha_agent/results.jsonl       # every evaluated alpha and its metrics/checks
.alpha_agent/state.json          # latest aggregate state
.alpha_agent/agent_runs/*.json   # complete per-run reports and planner events
.alpha_agent/submissions.jsonl   # successful submission records
```

The work directory is ignored by Git. Keep it if you want to retain history; changing `ALPHA_AGENT_WORKDIR` starts a separate result history.

### 5. Optional Streamlit console

```bash
python -m streamlit run streamlit_app.py
```

The console is intended for interactive local use. For a headless Linux machine, the `wqagent` wrapper and JSONL artifacts are usually simpler.

## Environment Variables

Copy from `.env.example` and set locally (never commit secrets):

```powershell
$env:WQB_EMAIL="your_email@example.com"
$env:WQB_PASSWORD="your_password"
# or
$env:WQB_COOKIE_HEADER="sessionid=...; csrftoken=..."

$env:ALPHA_AGENT_PLANNER_PROVIDER="heuristic"
$env:ALPHA_AGENT_PLANNER_MODEL="gpt5.5"
$env:ALPHA_AGENT_PLANNER_BASE_URL="https://api.openai.com/v1"
$env:ALPHA_AGENT_PLANNER_API_KEY_ENV="OPENAI_API_KEY"
$env:OPENAI_API_KEY="..."
```

The CLI and Streamlit app load `.env` automatically on startup. Sidebar fields are prefilled from `.env` when values are present; otherwise they start from safe defaults or blank credential fields.

## Quick Start

### Run the agent from CLI

```powershell
python .\alpha_research_agent.py --pretty run --budget 16 --max-iterations 10
```

Quant-style tuning example:

```powershell
python .\alpha_research_agent.py --pretty run --budget 20 --refine-top-k 10 --robustness-top-k 4 --robustness-score-threshold 550 --max-family-budget-share 0.4 --min-expression-novelty 0.12
```

Focus on a family:

```powershell
python .\alpha_research_agent.py --pretty run --family social_buzz --budget 12
```

Start from the Alpha101 replication library:

```powershell
python .\alpha_research_agent.py --pretty --idea-library .\alpha101_ideas.json run --family alpha101 --budget 24 --max-iterations 8
```

Use OpenAI planner:

```powershell
python .\alpha_research_agent.py --pretty --planner-provider openai run --budget 16
```

Manual submit mode (requires terminal approval):

```powershell
python .\alpha_research_agent.py --pretty run --submission-mode manual --interactive-approval
```

### Show current leaderboard

```powershell
python .\alpha_research_agent.py --pretty leaderboard --limit 10
```

### Run baseline-vs-agent evaluation

```powershell
python .\alpha_research_agent.py --pretty evaluate --cases .\docs\eval_cases.json
```

Output defaults to:

- `<workdir>/evaluation/report.json`

### Run the Presentation Frontend

For a classroom or pre demo, start with the frontend. It opens with a built-in showcase case, so it works even without a live WorldQuant session:

```powershell
python -m streamlit run .\streamlit_app.py
```

Then open the local URL printed by Streamlit, usually:

- `http://localhost:8501`

Recommended presentation flow:

1. Open the app and keep the built-in showcase loaded.
2. Walk through `Overview` for the executive result, readiness funnel, stage machine, and convergence curve.
3. Use `Agent Trace` to show planner actions, hypotheses, failure pressure, family frontier, and candidate details.
4. Use `Economic Logic` to explain the constrained optimization view and adjust cost/value assumptions live.
5. Use `Architecture` to explain design choices and show every hot-modifiable run parameter.
6. Optionally use `Run Live Agent` from the sidebar only after configuring credentials or a cookie header.

The presentation console provides:

- built-in showcase case for fast, reliable presentation
- idea library presets for the core search library and the Alpha101 replication library
- full live controls for auth, LLM planner, budget, family filters, novelty, robustness, retries, polling, workdir, and submission governance
- visual analytics for best score, quality-ready count, correlation-blocked candidates, submit-ready candidates, stage transitions, failure pressure, and family frontier
- candidate inspector with expression, settings, metrics, and check summary
- economic derivation panel showing alpha discovery as constrained expected utility maximization
- raw JSON artifact viewer and downloadable report

You can also load a saved run report from the sidebar with either:

- `Load Latest Workdir Report`
- `Run report JSON path`

More presentation notes are in [`docs/PRESENTATION_FRONTEND.md`](docs/PRESENTATION_FRONTEND.md).
Alpha101 replication notes are in [`docs/ALPHA101.md`](docs/ALPHA101.md).

## Basic Tests

```powershell
python -m unittest tests\test_planner.py tests\test_research_logic.py tests\test_progress.py tests\test_alpha101_library.py
```

## Reproducibility Artifacts

In agent workdir (`.alpha_agent` by default):

- `results.jsonl`: evaluated candidate records
- `submissions.jsonl`: submit attempts
- `state.json`: rolling summary
- `agent_runs/*.json`: full per-run reports with planner decisions and event logs
- `evaluation/report.json`: baseline-vs-agent comparison report

## Legacy Baseline Commands

Baseline scripts remain available:

```powershell
python .\alpha_research_pipeline.py --pretty search --budget 24 --seed-fraction 0.7
python .\alpha_research_pipeline.py --pretty leaderboard --limit 10
python .\alpha_research_pipeline.py --pretty submit-best
```

## Safety Notes

- Never commit credentials, cookies, keys, or private account data.
- Use `.env` and ignored local files for secrets.
- Keep `submission_mode=disabled` for research unless you intentionally enable stronger modes.
- Prefer manual approval for live demos and classroom evaluation.

## Disclaimer

Use responsibly and only with authorized credentials and permissions. Platform behavior and available endpoints may change over time.

## License

[MIT](LICENSE)
