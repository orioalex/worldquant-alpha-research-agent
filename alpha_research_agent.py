#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import alpha_research_pipeline as pipeline
from alpha_agent.config import AgentConfig, AgentRuntimeConfig, AuthConfig, ModelConfig
from alpha_agent.engine import AlphaResearchAgent, ResearchToolbox
from alpha_agent.evaluation import run_evaluation_suite
from alpha_agent.planner import HeuristicPlanner, OpenAIJsonPlanner
from local_env import load_local_dotenv
from worldquant_brain_cli import API_BASE, BrainApiError, DEFAULT_MAX_WAIT, DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT


DEFAULT_AGENT_WORKDIR = Path(".alpha_agent")
DEFAULT_EVAL_CASES = Path("docs") / "eval_cases.json"

load_local_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha Research Agent with planner-driven orchestration, tool use, and guarded submission flow."
    )
    add_auth_args(parser)
    add_common_agent_args(parser)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full alpha research agent loop.")
    add_run_args(run_parser)
    run_parser.set_defaults(func=command_run)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show current leaderboard from workdir.")
    leaderboard_parser.add_argument("--limit", type=int, default=10)
    leaderboard_parser.set_defaults(func=command_leaderboard)

    eval_parser = subparsers.add_parser("evaluate", help="Run baseline vs agent evaluation suite.")
    eval_parser.add_argument(
        "--cases",
        default=str(DEFAULT_EVAL_CASES),
        help="Path to evaluation JSON cases list.",
    )
    eval_parser.add_argument(
        "--output",
        default="",
        help="Optional output report path. Defaults to <workdir>/evaluation/report.json",
    )
    eval_parser.set_defaults(func=command_evaluate)

    return parser.parse_args()


def add_auth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--email", default=os.getenv("WQB_EMAIL"), help="BRAIN account email.")
    parser.add_argument("--password", default=os.getenv("WQB_PASSWORD"), help="BRAIN account password.")
    parser.add_argument(
        "--cookie-header",
        default=os.getenv("WQB_COOKIE_HEADER"),
        help="Optional raw Cookie header. If set, email/password login is skipped.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("WQB_API_BASE", API_BASE),
        help=f"API base URL. Defaults to {API_BASE}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("WQB_TIMEOUT", DEFAULT_TIMEOUT)),
        help="Single request timeout in seconds.",
    )


def add_common_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--idea-library",
        default=os.getenv("ALPHA_AGENT_IDEA_LIBRARY", str(pipeline.DEFAULT_IDEA_LIBRARY)),
        help="Path to idea library JSON.",
    )
    parser.add_argument(
        "--fields-summary",
        default=os.getenv("ALPHA_AGENT_FIELDS_SUMMARY", str(pipeline.DEFAULT_FIELDS_SUMMARY)),
        help="Path to cached fields summary JSON.",
    )
    parser.add_argument(
        "--workdir",
        default=os.getenv("ALPHA_AGENT_WORKDIR", str(DEFAULT_AGENT_WORKDIR)),
        help="Agent working directory for results/state/logs.",
    )
    parser.add_argument(
        "--planner-provider",
        choices=("heuristic", "openai"),
        default=os.getenv("ALPHA_AGENT_PLANNER_PROVIDER", "heuristic"),
        help="Planning backend.",
    )
    parser.add_argument(
        "--planner-model",
        default=os.getenv("ALPHA_AGENT_PLANNER_MODEL", "gpt5.5"),
        help="Model name for planner provider.",
    )
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_PLANNER_TEMPERATURE", "0.1")),
        help="Planner sampling temperature.",
    )
    parser.add_argument(
        "--planner-base-url",
        default=os.getenv("ALPHA_AGENT_PLANNER_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL for planner provider.",
    )
    parser.add_argument(
        "--planner-api-key-env",
        default=os.getenv("ALPHA_AGENT_PLANNER_API_KEY_ENV", "OPENAI_API_KEY"),
        help="Environment variable name that stores planner API key.",
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget", type=int, default=int(os.getenv("ALPHA_AGENT_BUDGET", "24")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("ALPHA_AGENT_MAX_ITERATIONS", "12")))
    parser.add_argument("--seed-fraction", type=float, default=float(os.getenv("ALPHA_AGENT_SEED_FRACTION", "0.7")))
    parser.add_argument(
        "--factor-pool-enabled",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ALPHA_AGENT_FACTOR_POOL_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"},
        help="Use bounded cross-family factor-pool combinations for generated seeds.",
    )
    parser.add_argument(
        "--factor-pool-max-atoms",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_FACTOR_POOL_MAX_ATOMS", "160")),
    )
    parser.add_argument(
        "--factor-pool-max-pairs",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_FACTOR_POOL_MAX_PAIRS", "240")),
    )
    parser.add_argument(
        "--factor-pool-max-triples",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_FACTOR_POOL_MAX_TRIPLES", "240")),
    )
    parser.add_argument(
        "--factor-pool-max-atom-reuse",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_FACTOR_POOL_MAX_ATOM_REUSE", "3")),
        help="Maximum number of factor-pool candidates that may reuse one atom.",
    )
    parser.add_argument(
        "--factor-pool-social-buzz-share",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_FACTOR_POOL_SOCIAL_BUZZ_SHARE", "0.5")),
        help="Relative sampling quota for the social_buzz family.",
    )
    parser.add_argument("--refine-top-k", type=int, default=int(os.getenv("ALPHA_AGENT_REFINE_TOP_K", "8")))
    parser.add_argument(
        "--robustness-top-k",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_ROBUSTNESS_TOP_K", "3")),
    )
    parser.add_argument(
        "--robustness-score-threshold",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_ROBUSTNESS_SCORE_THRESHOLD", "500")),
    )
    parser.add_argument("--family", action="append", default=[], help="Optional family filter (repeatable).")
    parser.add_argument(
        "--max-family-budget-share",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_MAX_FAMILY_BUDGET_SHARE", "0.45")),
    )
    parser.add_argument(
        "--no-improvement-batch-limit",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_NO_IMPROVEMENT_BATCH_LIMIT", "2")),
        help="Force a family diversification pivot after this many non-improving batches.",
    )
    parser.add_argument(
        "--min-expression-novelty",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_MIN_EXPRESSION_NOVELTY", "0.10")),
    )
    parser.add_argument(
        "--llm-expression-enabled",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ALPHA_AGENT_LLM_EXPRESSION_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"},
        help="Allow the OpenAI-compatible planner to propose complete FASTEXPR formulas.",
    )
    parser.add_argument(
        "--llm-expression-max-factors",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_LLM_EXPRESSION_MAX_FACTORS", "3")),
    )
    parser.add_argument(
        "--llm-expression-max-depth",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_LLM_EXPRESSION_MAX_DEPTH", "8")),
    )
    parser.add_argument(
        "--llm-expression-max-length",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_LLM_EXPRESSION_MAX_LENGTH", "800")),
    )
    parser.add_argument(
        "--force-diversify-on-correlation",
        action="store_true",
        default=os.getenv("ALPHA_AGENT_FORCE_DIVERSIFY_ON_CORRELATION", "false").lower()
        in {"1", "true", "yes", "on"},
        help="Skip local refinements and switch families when correlation checks block the frontier.",
    )
    parser.add_argument("--shuffle-seeds", action="store_true", default=True, help="Shuffle generated seeds.")
    parser.add_argument("--no-shuffle-seeds", action="store_false", dest="shuffle_seeds")
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("ALPHA_AGENT_RANDOM_SEED", "7")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("ALPHA_AGENT_RETRIES", "2")))
    parser.add_argument("--sleep-between", type=float, default=float(os.getenv("ALPHA_AGENT_SLEEP_BETWEEN", "1.0")))
    parser.add_argument("--max-wait", type=float, default=float(os.getenv("WQB_MAX_WAIT", str(DEFAULT_MAX_WAIT))))
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("WQB_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))),
    )
    parser.add_argument("--allow-pending-checks", action="store_true")
    parser.add_argument(
        "--submission-mode",
        choices=("disabled", "manual", "auto_approved"),
        default="disabled",
        help="Submission governance mode.",
    )
    parser.add_argument(
        "--interactive-approval",
        action="store_true",
        help="Prompt approval in terminal when submission-mode=manual.",
    )


def command_run(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args)
    planner = build_planner(runtime.model)
    approval_callback = None
    if args.submission_mode == "manual" and args.interactive_approval:
        approval_callback = cli_manual_approval
    agent = AlphaResearchAgent(runtime, planner=planner, approval_callback=approval_callback)
    result = agent.run()
    return result.to_dict()


def command_leaderboard(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args, run_overrides={})
    toolbox = ResearchToolbox(runtime)
    leaderboard = [pipeline.compact_record(item) for item in toolbox.leaderboard(limit=args.limit)]
    return {
        "count": len(leaderboard),
        "workdir": str(runtime.agent.workdir),
        "leaderboard": leaderboard,
    }


def command_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args, run_overrides={})
    case_file = Path(args.cases)
    output_path = Path(args.output) if args.output else Path(runtime.agent.workdir) / "evaluation" / "report.json"
    return run_evaluation_suite(
        runtime=runtime,
        case_file=case_file,
        output_path=output_path,
    )


def build_runtime(args: argparse.Namespace, run_overrides: Optional[Dict[str, Any]] = None) -> AgentRuntimeConfig:
    auth = AuthConfig(
        email=args.email,
        password=args.password,
        cookie_header=args.cookie_header,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    model = ModelConfig(
        provider=args.planner_provider,
        model=args.planner_model,
        temperature=args.planner_temperature,
        base_url=args.planner_base_url,
        api_key_env=args.planner_api_key_env,
        timeout=float(os.getenv("ALPHA_AGENT_PLANNER_TIMEOUT", "60")),
    )
    base_agent = AgentConfig(
        budget=getattr(args, "budget", 24),
        max_iterations=getattr(args, "max_iterations", 12),
        seed_fraction=getattr(args, "seed_fraction", 0.7),
        refine_top_k=getattr(args, "refine_top_k", 8),
        robustness_top_k=getattr(args, "robustness_top_k", 3),
        robustness_score_threshold=getattr(args, "robustness_score_threshold", 500.0),
        family_filter=tuple(getattr(args, "family", []) or ()),
        max_family_budget_share=getattr(args, "max_family_budget_share", 0.45),
        no_improvement_batch_limit=getattr(args, "no_improvement_batch_limit", 2),
        min_expression_novelty=getattr(args, "min_expression_novelty", 0.10),
        llm_expression_enabled=getattr(args, "llm_expression_enabled", True),
        llm_expression_max_factors=getattr(args, "llm_expression_max_factors", 3),
        llm_expression_max_depth=getattr(args, "llm_expression_max_depth", 8),
        llm_expression_max_length=getattr(args, "llm_expression_max_length", 800),
        factor_pool_enabled=getattr(args, "factor_pool_enabled", True),
        factor_pool_max_atoms=getattr(args, "factor_pool_max_atoms", 160),
        factor_pool_max_pairs=getattr(args, "factor_pool_max_pairs", 240),
        factor_pool_max_triples=getattr(args, "factor_pool_max_triples", 240),
        factor_pool_max_atom_reuse=getattr(args, "factor_pool_max_atom_reuse", 3),
        factor_pool_social_buzz_share=getattr(args, "factor_pool_social_buzz_share", 0.5),
        force_diversify_on_correlation=getattr(args, "force_diversify_on_correlation", False),
        shuffle_seeds=getattr(args, "shuffle_seeds", True),
        random_seed=getattr(args, "random_seed", 7),
        retries=getattr(args, "retries", 2),
        sleep_between=getattr(args, "sleep_between", 1.0),
        max_wait=getattr(args, "max_wait", DEFAULT_MAX_WAIT),
        poll_interval=getattr(args, "poll_interval", DEFAULT_POLL_INTERVAL),
        allow_pending_checks=getattr(args, "allow_pending_checks", False),
        submission_mode=getattr(args, "submission_mode", "disabled"),
        workdir=Path(args.workdir),
        idea_library=Path(args.idea_library),
        fields_summary=Path(args.fields_summary),
    )
    if run_overrides:
        base_agent = replace(base_agent, **run_overrides)
    return AgentRuntimeConfig(auth=auth, model=model, agent=base_agent)


def build_planner(model_config: ModelConfig) -> Any:
    if model_config.provider == "openai":
        return OpenAIJsonPlanner(model_config=model_config)
    return HeuristicPlanner()


def cli_manual_approval(candidate: Dict[str, Any]) -> bool:
    alpha_id = candidate.get("alpha_id")
    score = candidate.get("score")
    family = candidate.get("family")
    prompt = f"Approve submit for alpha {alpha_id} (family={family}, score={score})? [y/N]: "
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def render_output(payload: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    try:
        payload = args.func(args)
        render_output(payload, pretty=args.pretty)
        return 0
    except BrainApiError as exc:
        message = str(exc)
        if exc.status is not None:
            message = f"{message} [status={exc.status}]"
        if exc.url:
            message = f"{message} [url={exc.url}]"
        print(message, file=sys.stderr)
        if exc.payload is not None:
            print(json.dumps(exc.payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
