from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import alpha_research_pipeline as pipeline
from alpha_agent.config import AgentRuntimeConfig
from alpha_agent.planner import HeuristicPlanner, PlannerAction
from alpha_agent.research_logic import (
    ResearchNotebook,
    build_check_aware_refinements,
    build_correlation_repair_candidates,
    build_robustness_candidates,
    dedupe_candidates,
    safe_float,
)
from worldquant_brain_cli import BrainApiError, BrainClient


ApprovalCallback = Callable[[Dict[str, Any]], bool]
ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class AgentRunResult:
    run_id: str
    started_at: str
    finished_at: str
    report_path: Path
    summary: Dict[str, Any]
    leaderboard: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    submissions: List[Dict[str, Any]]
    state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "report_path": str(self.report_path),
            "summary": self.summary,
            "leaderboard": self.leaderboard,
            "events": self.events,
            "submissions": self.submissions,
            "state": self.state,
        }


class ResearchToolbox:
    def __init__(self, runtime: AgentRuntimeConfig) -> None:
        self.runtime = runtime
        self.agent_cfg = runtime.agent
        self.workdir = Path(self.agent_cfg.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.results_store = pipeline.JsonlStore(self.workdir / "results.jsonl")
        self.submissions_store = pipeline.JsonlStore(self.workdir / "submissions.jsonl")
        self.state_path = self.workdir / "state.json"
        self.agent_runs_dir = self.workdir / "agent_runs"
        self.agent_runs_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[BrainClient] = None
        self.library = pipeline.load_json(Path(self.agent_cfg.idea_library))
        self.available_fields = pipeline.load_available_fields(str(self.agent_cfg.fields_summary))

        self.results: List[Dict[str, Any]] = []
        self.submissions: List[Dict[str, Any]] = []
        self.evaluated_signatures: set[str] = set()
        self.submitted_alpha_ids: set[str] = set()
        self.seed_queue: List[pipeline.Candidate] = []
        self.all_ordered_seeds: List[pipeline.Candidate] = []
        self.refresh_records()
        self.refresh_seed_queue()

    def _build_client(self) -> BrainClient:
        auth = self.runtime.auth
        client = BrainClient(
            base_url=auth.base_url,
            timeout=auth.timeout,
            cookie_header=auth.cookie_header,
        )
        client.opener.addheaders = [("User-Agent", pipeline.DEFAULT_USER_AGENT)]
        if auth.cookie_header:
            return client
        if not auth.email or not auth.password:
            raise BrainApiError("Missing credentials. Set email/password or cookie header.")
        client.login(auth.email, auth.password)
        return client

    @property
    def client(self) -> BrainClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def refresh_records(self) -> None:
        self.results = self.results_store.read_all()
        self.submissions = self.submissions_store.read_all()
        self.evaluated_signatures = {
            signature
            for signature in (pipeline.record_signature(record) for record in self.results)
            if signature
        }
        self.submitted_alpha_ids = {
            str(record.get("alpha_id"))
            for record in self.submissions
            if isinstance(record.get("alpha_id"), str)
        }

    def refresh_seed_queue(self) -> int:
        pivot_families = pipeline.correlation_pivot_families(self.results)
        seeds = pipeline.generate_seed_candidates(
            library=self.library,
            family_filter=set(self.agent_cfg.family_filter),
            available_fields=self.available_fields,
        )
        if self.agent_cfg.force_diversify_on_correlation and pivot_families:
            seeds = [candidate for candidate in seeds if candidate.family not in pivot_families]
        ordered = pipeline.prioritize_seed_candidates(
            seeds=seeds,
            blocked_families=pivot_families,
            shuffle_generated=self.agent_cfg.shuffle_seeds,
            random_seed=self.agent_cfg.random_seed,
        )
        self.all_ordered_seeds = ordered
        self.seed_queue = [
            candidate
            for candidate in ordered
            if candidate.signature() not in self.evaluated_signatures
        ]
        return len(self.seed_queue)

    def expression_field_catalog(self, limit: int = 240) -> List[str]:
        """Return a compact, useful field catalog for the planner prompt."""
        if not self.available_fields:
            return []
        preferred_names = [
            "close",
            "open",
            "high",
            "low",
            "vwap",
            "volume",
            "returns",
            "market_cap",
            "scl12_buzz",
        ]
        library_fields: List[str] = []
        for family in self.library.get("families", []):
            if not isinstance(family, dict):
                continue
            library_fields.extend(
                str(item)
                for item in family.get("fields", [])
                if isinstance(item, str)
            )
        ordered: List[str] = []
        seen: set[str] = set()
        for field in preferred_names + library_fields + sorted(self.available_fields):
            if field in self.available_fields and field not in seen:
                seen.add(field)
                ordered.append(field)
                if len(ordered) >= max(1, int(limit)):
                    break
        return ordered

    def pop_seed_candidates(self, count: int, *, focus_family: Optional[str] = None) -> List[pipeline.Candidate]:
        count = max(0, int(count))
        if count == 0:
            return []
        if focus_family:
            focused = [candidate for candidate in self.seed_queue if candidate.family == focus_family]
            if focused:
                selected = focused[:count]
                selected_signatures = {candidate.signature() for candidate in selected}
                self.seed_queue = [
                    candidate
                    for candidate in self.seed_queue
                    if candidate.signature() not in selected_signatures
                ]
                return selected
        batch = self.seed_queue[:count]
        self.seed_queue = self.seed_queue[count:]
        return batch

    def leaderboard(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        return pipeline.build_leaderboard(
            self.results,
            family_filter=set(self.agent_cfg.family_filter),
            limit=limit,
        )

    def preview_refine_candidates(
        self,
        *,
        frontier_records: Sequence[Dict[str, Any]],
    ) -> Tuple[List[pipeline.Candidate], set[str], List[Dict[str, Any]]]:
        top_records = list(frontier_records[: self.agent_cfg.refine_top_k])
        blocked_families = pipeline.correlation_pivot_families(top_records)
        generic = pipeline.build_refinement_candidates(
            top_records,
            family_filter=set(self.agent_cfg.family_filter),
            blocked_families=blocked_families,
        )
        correlation_repairs = build_correlation_repair_candidates(
            top_records,
            family_filter=set(self.agent_cfg.family_filter),
        )
        check_aware = build_check_aware_refinements(
            top_records,
            family_filter=set(self.agent_cfg.family_filter),
            blocked_families=blocked_families,
        )
        correlation_blocked = bool(blocked_families) and any(
            pipeline.should_pivot_away_from_record(record) for record in top_records
        )
        if self.agent_cfg.force_diversify_on_correlation and correlation_blocked:
            merged = []
        else:
            merged = dedupe_candidates(correlation_repairs + check_aware + generic)
        fresh = [
            candidate
            for candidate in merged
            if candidate.signature() not in self.evaluated_signatures
        ]
        return fresh, blocked_families, top_records

    def preview_diversification_candidates(self, blocked_families: set[str]) -> List[pipeline.Candidate]:
        return pipeline.build_diversification_candidates(
            seeds=self.all_ordered_seeds,
            excluded_signatures=self.evaluated_signatures,
            blocked_families=blocked_families,
        )

    def preview_robustness_candidates(
        self,
        *,
        frontier_records: Sequence[Dict[str, Any]],
    ) -> List[pipeline.Candidate]:
        candidates = build_robustness_candidates(
            frontier_records,
            family_filter=set(self.agent_cfg.family_filter),
            top_k=self.agent_cfg.robustness_top_k,
        )
        return [
            candidate
            for candidate in candidates
            if candidate.signature() not in self.evaluated_signatures
        ]

    def evaluate_candidates(
        self,
        candidates: Sequence[pipeline.Candidate],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        submission_attempts: List[Dict[str, Any]] = []

        def emit_progress(payload: Dict[str, Any]) -> None:
            if payload.get("type") == "candidate_completed":
                self.refresh_records()
                payload["leaderboard"] = [
                    pipeline.compact_record(item)
                    for item in self.leaderboard(limit=8)
                ]
                payload["state"] = self.write_state()
            if progress_callback:
                progress_callback(payload)

        evaluated = pipeline.evaluate_batch(
            client=self.client,
            candidates=candidates,
            results_store=self.results_store,
            submissions_store=self.submissions_store,
            submitted_alpha_ids=self.submitted_alpha_ids,
            max_wait=self.agent_cfg.max_wait,
            poll_interval=self.agent_cfg.poll_interval,
            retries=self.agent_cfg.retries,
            sleep_between=self.agent_cfg.sleep_between,
            should_attempt_submit=False,
            allow_pending_checks=self.agent_cfg.allow_pending_checks,
            stop_on_submittable=False,
            submission_attempts=submission_attempts,
            progress_callback=emit_progress,
        )
        self.refresh_records()
        return evaluated

    def best_submittable_candidate(self, *, target_alpha_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        allow_pending = self.agent_cfg.allow_pending_checks
        for record in self.leaderboard(limit=200):
            alpha_id = record.get("alpha_id")
            if not isinstance(alpha_id, str):
                continue
            if alpha_id in self.submitted_alpha_ids:
                continue
            if target_alpha_id and alpha_id != target_alpha_id:
                continue
            if not pipeline.record_is_submit_ready(record, allow_pending_checks=allow_pending):
                continue
            return record
        return None

    def submit_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        alpha_id = str(record["alpha_id"])
        submission = pipeline.attempt_submit(
            client=self.client,
            alpha_id=alpha_id,
            record=record,
            submissions_store=self.submissions_store,
            allow_pending_checks=self.agent_cfg.allow_pending_checks,
            max_wait=self.agent_cfg.max_wait,
            poll_interval=self.agent_cfg.poll_interval,
        )
        self.refresh_records()
        return submission

    def write_state(self) -> Dict[str, Any]:
        leaderboard = self.leaderboard(limit=10)
        return pipeline.write_state(
            self.state_path,
            self.results,
            self.submissions,
            leaderboard,
        )


class AlphaResearchAgent:
    def __init__(
        self,
        runtime: AgentRuntimeConfig,
        planner: Optional[Any] = None,
        approval_callback: Optional[ApprovalCallback] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.runtime = runtime
        self.planner = planner or HeuristicPlanner()
        self.toolbox = ResearchToolbox(runtime)
        self.approval_callback = approval_callback
        self.progress_callback = progress_callback
        self.events: List[Dict[str, Any]] = []
        self.submission_attempts: List[Dict[str, Any]] = []
        self.attempted_submission_ids: set[str] = set()
        self.seed_evaluated_count = 0
        self.no_improvement_batches = 0
        self.last_batch_families: set[str] = set()
        self.notebook = ResearchNotebook(
            budget=max(1, int(runtime.agent.budget)),
            max_family_budget_share=runtime.agent.max_family_budget_share,
            min_expression_novelty=runtime.agent.min_expression_novelty,
            robustness_score_threshold=runtime.agent.robustness_score_threshold,
        )

    def run(self) -> AgentRunResult:
        started_at = pipeline.iso_now()
        run_id = self._build_run_id(started_at)
        budget = max(0, int(self.runtime.agent.budget))
        seed_target = (
            min(budget, max(1, int(math.ceil(budget * float(self.runtime.agent.seed_fraction)))))
            if budget > 0
            else 0
        )
        evaluated_total = 0
        evaluated_this_run = 0
        stop_reason = "Completed."
        iteration_executed = 0
        current_stage = "explore"
        self._emit_progress(
            {
                "type": "run_started",
                "run_id": run_id,
                "started_at": started_at,
                "budget": budget,
                "max_iterations": self.runtime.agent.max_iterations,
                "workdir": str(self.runtime.agent.workdir),
            }
        )

        for iteration in range(1, self.runtime.agent.max_iterations + 1):
            iteration_executed = iteration
            remaining_budget = budget - evaluated_total
            if remaining_budget <= 0:
                stop_reason = "Budget exhausted."
                break

            self.toolbox.refresh_seed_queue()
            frontier = self.toolbox.leaderboard(limit=max(20, self.runtime.agent.refine_top_k))
            current_stage, stage_reason = self.notebook.determine_stage(
                iteration=iteration,
                evaluated_total=evaluated_total,
                seed_target=seed_target,
                leaderboard=frontier,
                records=self.toolbox.results,
            )
            refine_candidates, blocked_families, _ = self.toolbox.preview_refine_candidates(
                frontier_records=frontier
            )
            diversify_candidates = self.toolbox.preview_diversification_candidates(blocked_families)
            robustness_candidates = self.toolbox.preview_robustness_candidates(frontier_records=frontier)

            ranked_refine = self.notebook.rank_candidates(
                candidates=refine_candidates,
                frontier_records=frontier,
                stage=current_stage,
                blocked_families=blocked_families,
            )
            ranked_diversify = self.notebook.rank_candidates(
                candidates=diversify_candidates,
                frontier_records=frontier,
                stage=current_stage,
                blocked_families=blocked_families,
            )
            ranked_robustness = self.notebook.rank_candidates(
                candidates=robustness_candidates,
                frontier_records=frontier,
                stage="robustness",
                blocked_families=set(),
            )

            context = self._build_context(
                iteration=iteration,
                budget=budget,
                evaluated_total=evaluated_total,
                seed_target=seed_target,
                stage=current_stage,
                stage_reason=stage_reason,
                refine_candidates=ranked_refine,
                diversify_candidates=ranked_diversify,
                robustness_candidates=ranked_robustness,
                frontier=frontier,
            )
            decision = self._decide(context).clamped(remaining_budget)
            self.notebook.record_hypothesis(
                iteration=iteration,
                stage=current_stage,
                action=decision.action,
                hypothesis=decision.hypothesis or "No explicit hypothesis provided.",
                rationale=decision.rationale,
                focus_family=decision.focus_family,
                risk_note=decision.risk_note,
            )
            self._emit_progress(
                {
                    "type": "planner_decision",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": current_stage,
                    "stage_reason": stage_reason,
                    "decision": decision.raw or {},
                    "action": decision.action,
                    "batch_size": decision.batch_size,
                    "rationale": decision.rationale,
                    "hypothesis": decision.hypothesis,
                    "risk_note": decision.risk_note,
                    "context": context,
                    "leaderboard": [pipeline.compact_record(item) for item in frontier[:8]],
                }
            )

            if decision.action == "stop":
                stop_reason = decision.rationale or "Planner requested stop."
                self._append_event(
                    iteration=iteration,
                    stage=current_stage,
                    action="stop",
                    rationale=stop_reason,
                    details={"planner_context": context},
                    hypothesis=decision.hypothesis,
                    risk_note=decision.risk_note,
                )
                break

            if decision.action == "submit_best":
                submission_outcome = self._execute_submit_action(decision, iteration, current_stage)
                if submission_outcome is not None:
                    self.submission_attempts.append(submission_outcome)
                self.toolbox.write_state()
                continue

            batch: List[pipeline.Candidate] = []
            expression_error: Optional[str] = None
            if decision.action == "propose_expression":
                candidate, expression_error = self._build_llm_expression_candidate(
                    decision=decision,
                    iteration=iteration,
                    stage=current_stage,
                )
                if candidate is not None:
                    batch = self._select_batch(
                        candidates=[candidate],
                        batch_size=1,
                        stage=current_stage,
                        frontier=frontier,
                        blocked_families=blocked_families,
                        focus_family=None,
                    )
            elif decision.action == "evaluate_seed":
                raw_seed_batch = self.toolbox.pop_seed_candidates(
                    max(decision.batch_size * 2, decision.batch_size),
                    focus_family=decision.focus_family,
                )
                batch = self._select_batch(
                    candidates=raw_seed_batch,
                    batch_size=decision.batch_size,
                    stage=current_stage,
                    frontier=frontier,
                    blocked_families=blocked_families,
                    focus_family=decision.focus_family,
                )
            elif decision.action == "evaluate_refine":
                batch = self._select_batch(
                    candidates=ranked_refine,
                    batch_size=decision.batch_size,
                    stage=current_stage,
                    frontier=frontier,
                    blocked_families=blocked_families,
                    focus_family=decision.focus_family,
                )
            elif decision.action == "evaluate_diversify":
                batch = self._select_batch(
                    candidates=ranked_diversify,
                    batch_size=decision.batch_size,
                    stage=current_stage,
                    frontier=frontier,
                    blocked_families=blocked_families,
                    focus_family=decision.focus_family,
                )
            elif decision.action == "evaluate_robustness":
                batch = self._select_batch(
                    candidates=ranked_robustness,
                    batch_size=decision.batch_size,
                    stage="robustness",
                    frontier=frontier,
                    blocked_families=set(),
                    focus_family=decision.focus_family,
                    enforce_family_cap=False,
                )

            if not batch:
                stop_reason = f"No candidates available for {decision.action}."
                self._append_event(
                    iteration=iteration,
                    stage=current_stage,
                    action=decision.action,
                    rationale=decision.rationale,
                    details={
                        "requested_batch": decision.batch_size,
                        "executed_batch": 0,
                        "expression_error": expression_error,
                    },
                    hypothesis=decision.hypothesis,
                    risk_note=decision.risk_note,
                )
                continue

            self._emit_progress(
                {
                    "type": "batch_started",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": current_stage,
                    "action": decision.action,
                    "batch_size": len(batch),
                    "candidates": [candidate.to_record() for candidate in batch],
                    "remaining_budget": remaining_budget,
                }
            )

            records = self.toolbox.evaluate_candidates(
                batch,
                progress_callback=lambda payload, iteration=iteration, stage=current_stage, action=decision.action: self._emit_progress(
                    {
                        **payload,
                        "run_id": run_id,
                        "iteration": iteration,
                        "stage": stage,
                        "action": action,
                        "budget": budget,
                    }
                ),
            )
            self.notebook.observe_records(records)
            evaluated_total += len(records)
            evaluated_this_run += len(records)
            if decision.action == "evaluate_seed":
                self.seed_evaluated_count += len(records)

            best_score_before = safe_float(frontier[0].get("score")) if frontier else None
            best_score_after = self._best_score()
            improved = (
                best_score_after is not None
                and (best_score_before is None or best_score_after > best_score_before + 1e-9)
            )
            if improved:
                self.no_improvement_batches = 0
            else:
                self.no_improvement_batches += 1
            self.last_batch_families = {
                str(record.get("family"))
                for record in records
                if isinstance(record.get("family"), str) and record.get("family")
            }

            event_details = {
                "requested_batch": decision.batch_size,
                "executed_batch": len(records),
                "ok_count": sum(1 for record in records if record.get("status") == "ok"),
                "error_count": sum(1 for record in records if record.get("status") == "error"),
                "quality_ready_count": sum(1 for record in records if pipeline.record_quality_ready(record)),
                "submittable_count": sum(1 for record in records if pipeline.record_is_submit_ready(record)),
                "correlation_blocked_count": sum(
                    1
                    for record in records
                    if pipeline.record_quality_ready(record) and record.get("failed_correlation_checks")
                ),
                "families": sorted({str(record.get("family") or "") for record in records}),
                "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results, top_k=5),
                "best_score_before": best_score_before,
                "best_score_after": best_score_after,
                "best_alpha_after": self._best_alpha_id(),
                "improved": improved,
                "no_improvement_batches": self.no_improvement_batches,
                "last_batch_families": sorted(self.last_batch_families),
            }
            self._append_event(
                iteration=iteration,
                stage=current_stage,
                action=decision.action,
                rationale=decision.rationale,
                details=event_details,
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            self.toolbox.write_state()
            self._emit_progress(
                {
                    "type": "batch_completed",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": current_stage,
                    "action": decision.action,
                    "details": event_details,
                    "leaderboard": [
                        pipeline.compact_record(item)
                        for item in self.toolbox.leaderboard(limit=8)
                    ],
                    "state": self.toolbox.write_state(),
                }
            )

        finished_at = pipeline.iso_now()
        leaderboard = [pipeline.compact_record(item) for item in self.toolbox.leaderboard(limit=10)]
        state = self.toolbox.write_state()
        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "budget": budget,
            "evaluated_count": len(self.toolbox.results),
            "evaluated_this_run": evaluated_this_run,
            "seed_evaluated_this_run": self.seed_evaluated_count,
            "submission_attempts_this_run": len(self.submission_attempts),
            "best_score": self._best_score(),
            "best_alpha_id": self._best_alpha_id(),
            "quality_ready_count": sum(1 for record in self.toolbox.results if pipeline.record_quality_ready(record)),
            "submittable_count": sum(
                1 for record in self.toolbox.results if pipeline.record_is_submit_ready(record)
            ),
            "correlation_blocked_count": sum(
                1
                for record in self.toolbox.results
                if pipeline.record_quality_ready(record) and record.get("failed_correlation_checks")
            ),
            "iterations_executed": iteration_executed,
            "stop_reason": stop_reason,
            "submission_mode": self.runtime.agent.submission_mode,
            "final_stage": current_stage,
            "stage_history": self.notebook.stage_history,
            "family_stats": self.notebook.family_stats(self.toolbox.results),
            "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results),
            "hypothesis_log_tail": self.notebook.recent_hypotheses(limit=8),
        }
        report_path = self._write_run_report(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
            leaderboard=leaderboard,
            state=state,
        )
        result = AgentRunResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            report_path=report_path,
            summary=summary,
            leaderboard=leaderboard,
            events=self.events,
            submissions=self.submission_attempts,
            state=state,
        )
        self._emit_progress({"type": "run_finished", "run_id": run_id, "result": result.to_dict()})
        return result

    def _decide(self, context: Dict[str, Any]) -> PlannerAction:
        action = self.planner.decide(context)
        if action.action not in {
            "evaluate_seed",
            "evaluate_refine",
            "evaluate_diversify",
            "evaluate_robustness",
            "propose_expression",
            "submit_best",
            "stop",
        }:
            return PlannerAction.stop(f"Invalid planner action: {action.action}")
        no_improvement_batches = int(context.get("no_improvement_batches") or 0)
        diversification_available = int(context.get("diversification_candidates_available") or 0)
        limit = max(1, int(self.runtime.agent.no_improvement_batch_limit))
        if (
            no_improvement_batches >= limit
            and diversification_available > 0
            and action.action != "submit_best"
        ):
            family = self._choose_diversification_family(context)
            return PlannerAction(
                action="evaluate_diversify",
                batch_size=max(1, action.batch_size or 1),
                rationale=(
                    f"Forced diversification after {no_improvement_batches} consecutive batches "
                    "without improving the best score."
                ),
                hypothesis="A different family should restore search progress after a stagnant frontier.",
                focus_family=family,
                risk_note="Guardrail overrode the planner to prevent repeated local refinement.",
                raw={
                    **(action.raw or {}),
                    "guardrail": "no_improvement_family_pivot",
                    "original_action": action.action,
                },
            )
        return action

    def _build_llm_expression_candidate(
        self,
        *,
        decision: PlannerAction,
        iteration: int,
        stage: str,
    ) -> Tuple[Optional[pipeline.Candidate], Optional[str]]:
        if not self.runtime.agent.llm_expression_enabled:
            return None, "LLM expression generation is disabled."
        expression = str(decision.expression or "").strip()
        if not expression:
            return None, "Planner returned propose_expression without expression."
        validation = pipeline.validate_llm_expression(
            expression,
            available_fields=self.toolbox.available_fields,
            max_factors=self.runtime.agent.llm_expression_max_factors,
            max_depth=self.runtime.agent.llm_expression_max_depth,
            max_length=self.runtime.agent.llm_expression_max_length,
        )
        if not validation["valid"]:
            return None, "; ".join(validation["errors"])
        candidate = pipeline.Candidate(
            expression=expression,
            settings=pipeline.normalize_settings({}),
            family="llm_generated",
            idea_name=f"llm_expression_{iteration:03d}",
            stage="llm_generated",
            priority=100.0,
            metadata={
                "llm_expression": True,
                "requested_family": decision.focus_family,
                "fields": validation["fields"],
                "functions": validation["functions"],
                "factor_count": len(validation["fields"]),
                "nesting_depth": validation["depth"],
            },
        )
        if candidate.signature() in self.toolbox.evaluated_signatures:
            return None, "LLM expression duplicates an already evaluated candidate."
        return candidate, None

    def _choose_diversification_family(self, context: Dict[str, Any]) -> Optional[str]:
        last_families = {
            str(family)
            for family in context.get("last_batch_families") or []
            if family
        }
        capped = {
            str(item.get("family"))
            for item in context.get("family_stats") or []
            if item.get("family") and item.get("family_cap_reached")
        }
        families = [
            str(family)
            for family in context.get("diversification_families") or []
            if family
        ]
        for family in families:
            if family not in last_families and family not in capped:
                return family
        for family in families:
            if family not in capped:
                return family
        return None

    def _execute_submit_action(
        self,
        decision: PlannerAction,
        iteration: int,
        stage: str,
    ) -> Optional[Dict[str, Any]]:
        mode = self.runtime.agent.submission_mode
        if stage != "harvest":
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": "blocked",
                    "reason": "Agent submissions require harvest stage after robustness evidence.",
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        candidate = self.toolbox.best_submittable_candidate(target_alpha_id=decision.target_alpha_id)
        if not candidate:
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={"result": "skipped", "reason": "No eligible submittable candidate."},
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        alpha_id = str(candidate.get("alpha_id"))
        if alpha_id in self.attempted_submission_ids:
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": "skipped",
                    "reason": "Submission was already attempted for this alpha in the current run.",
                    "alpha_id": alpha_id,
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        if mode == "disabled":
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": "blocked",
                    "reason": "Submission mode is disabled.",
                    "alpha_id": candidate.get("alpha_id"),
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        approved = mode == "auto_approved"
        if mode == "manual":
            approved = self.approval_callback(candidate) if self.approval_callback else False
        if not approved:
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": "blocked",
                    "reason": "Manual approval not granted.",
                    "alpha_id": candidate.get("alpha_id"),
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        try:
            self.attempted_submission_ids.add(alpha_id)
            submission = self.toolbox.submit_record(candidate)
            result = {
                "alpha_id": submission.get("alpha_id"),
                "result": submission.get("result", "ok"),
                "submitted_at": submission.get("submitted_at", pipeline.iso_now()),
                "response": submission.get("response"),
            }
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": submission.get("result", "ok"),
                    "alpha_id": submission.get("alpha_id"),
                    "message": (submission.get("response") or {}).get("message"),
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return result
        except BrainApiError as exc:
            error_payload = {
                "alpha_id": candidate.get("alpha_id"),
                "result": "error",
                "submitted_at": pipeline.iso_now(),
                "response": {
                    "message": str(exc),
                    "status": exc.status,
                    "url": exc.url,
                    "payload": exc.payload,
                },
            }
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={"result": "error", "alpha_id": candidate.get("alpha_id"), "message": str(exc)},
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return error_payload

    def _select_batch(
        self,
        *,
        candidates: Sequence[pipeline.Candidate],
        batch_size: int,
        stage: str,
        frontier: Sequence[Dict[str, Any]],
        blocked_families: set[str],
        focus_family: Optional[str] = None,
        enforce_family_cap: bool = True,
    ) -> List[pipeline.Candidate]:
        if not candidates:
            return []
        candidate_pool = list(candidates)
        if focus_family:
            focused = [candidate for candidate in candidate_pool if candidate.family == focus_family]
            if focused and (
                not enforce_family_cap
                or any(not self.notebook.is_family_capped(candidate.family) for candidate in focused)
            ):
                candidate_pool = focused
        ranked = self.notebook.rank_candidates(
            candidates=candidate_pool,
            frontier_records=frontier,
            stage=stage,
            blocked_families=blocked_families,
        )
        if not enforce_family_cap:
            return list(ranked[:batch_size])
        selected: List[pipeline.Candidate] = []
        selected_by_family: Dict[str, int] = {}
        for candidate in ranked:
            family = candidate.family
            if self.notebook.is_family_capped(family):
                continue
            if selected_by_family.get(family, 0) >= self.notebook.family_cap_remaining(family):
                continue
            selected.append(candidate)
            selected_by_family[family] = selected_by_family.get(family, 0) + 1
            if len(selected) >= batch_size:
                break
        return selected

    def _append_event(
        self,
        *,
        iteration: int,
        stage: str,
        action: str,
        rationale: str,
        details: Dict[str, Any],
        hypothesis: str,
        risk_note: str,
    ) -> None:
        event = {
                "timestamp": pipeline.iso_now(),
                "iteration": iteration,
                "stage": stage,
                "action": action,
                "rationale": rationale,
                "hypothesis": hypothesis,
                "risk_note": risk_note,
                "details": details,
        }
        self.events.append(event)
        self._emit_progress({"type": "event_appended", "event": event})

    def _emit_progress(self, payload: Dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        payload.setdefault("timestamp", pipeline.iso_now())
        try:
            self.progress_callback(payload)
        except Exception:
            # Progress rendering must not interrupt a live research run.
            pass

    def _build_context(
        self,
        *,
        iteration: int,
        budget: int,
        evaluated_total: int,
        seed_target: int,
        stage: str,
        stage_reason: str,
        refine_candidates: Sequence[pipeline.Candidate],
        diversify_candidates: Sequence[pipeline.Candidate],
        robustness_candidates: Sequence[pipeline.Candidate],
        frontier: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        remaining_budget = max(0, budget - evaluated_total)
        best = frontier[0] if frontier else {}
        best_submittable = self.toolbox.best_submittable_candidate()
        return {
            "iteration": iteration,
            "budget": budget,
            "evaluated_total": evaluated_total,
            "remaining_budget": remaining_budget,
            "seed_target": seed_target,
            "seed_target_remaining": max(0, seed_target - self.seed_evaluated_count),
            "seed_queue_remaining": len(self.toolbox.seed_queue),
            "refine_candidates_available": len(refine_candidates),
            "diversification_candidates_available": len(diversify_candidates),
            "robustness_candidates_available": len(robustness_candidates),
            "available_families": sorted(
                {
                    candidate.family
                    for candidate in (
                        list(self.toolbox.seed_queue)
                        + list(refine_candidates)
                        + list(diversify_candidates)
                        + list(robustness_candidates)
                    )
                    if candidate.family
                }
            ),
            "leaderboard_count": len(frontier),
            "best_score": best.get("score"),
            "best_alpha_id": best.get("alpha_id"),
            "best_submittable_alpha_id": best_submittable.get("alpha_id") if best_submittable else None,
            "submission_mode": self.runtime.agent.submission_mode,
            "allow_pending_checks": self.runtime.agent.allow_pending_checks,
            "family_filter": list(self.runtime.agent.family_filter),
            "research_stage": stage,
            "stage_reason": stage_reason,
            "family_budget_cap": self.notebook.family_cap(),
            "expression_generation_enabled": self.runtime.agent.llm_expression_enabled,
            "expression_constraints": {
                "max_factors": self.runtime.agent.llm_expression_max_factors,
                "max_depth": self.runtime.agent.llm_expression_max_depth,
                "max_length": self.runtime.agent.llm_expression_max_length,
            },
            "available_expression_fields": self.toolbox.expression_field_catalog(),
            "allowed_expression_operators": sorted(pipeline.WQ_ALLOWED_OPERATORS),
            "llm_expression_proposals": sum(
                1 for event in self.events if event.get("action") == "propose_expression"
            ),
            "no_improvement_batches": self.no_improvement_batches,
            "no_improvement_batch_limit": self.runtime.agent.no_improvement_batch_limit,
            "last_batch_families": sorted(self.last_batch_families),
            "diversification_families": sorted(
                {candidate.family for candidate in diversify_candidates if candidate.family}
            ),
            "family_stats": self.notebook.family_stats(self.toolbox.results),
            "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results),
            "recent_hypotheses": self.notebook.recent_hypotheses(limit=5),
            "recent_events": self.events[-4:],
            "frontier_preview": [
                {
                    "alpha_id": item.get("alpha_id"),
                    "family": item.get("family"),
                    "expression": item.get("expression"),
                    "score": item.get("score"),
                    "failed_checks": item.get("failed_checks"),
                    "stage": item.get("stage"),
                }
                for item in frontier[:5]
            ],
        }

    def _best_score(self) -> Optional[float]:
        leaderboard = self.toolbox.leaderboard(limit=1)
        if not leaderboard:
            return None
        return safe_float(leaderboard[0].get("score"))

    def _best_alpha_id(self) -> Optional[str]:
        leaderboard = self.toolbox.leaderboard(limit=1)
        if not leaderboard:
            return None
        alpha_id = leaderboard[0].get("alpha_id")
        return str(alpha_id) if alpha_id else None

    def _build_run_id(self, started_at: str) -> str:
        seed = json.dumps(
            {
                "started_at": started_at,
                "budget": self.runtime.agent.budget,
                "family_filter": list(self.runtime.agent.family_filter),
                "submission_mode": self.runtime.agent.submission_mode,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
        return f"agent_{digest}"

    def _write_run_report(
        self,
        *,
        run_id: str,
        started_at: str,
        finished_at: str,
        summary: Dict[str, Any],
        leaderboard: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Path:
        report = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime": {
                "auth": {
                    "base_url": self.runtime.auth.base_url,
                    "timeout": self.runtime.auth.timeout,
                    "using_cookie_header": bool(self.runtime.auth.cookie_header),
                    "email_configured": bool(self.runtime.auth.email),
                },
                "model": {
                    "provider": self.runtime.model.provider,
                    "model": self.runtime.model.model,
                    "temperature": self.runtime.model.temperature,
                    "base_url": self.runtime.model.base_url,
                },
                "agent": {
                    "budget": self.runtime.agent.budget,
                    "max_iterations": self.runtime.agent.max_iterations,
                    "seed_fraction": self.runtime.agent.seed_fraction,
                    "refine_top_k": self.runtime.agent.refine_top_k,
                    "robustness_top_k": self.runtime.agent.robustness_top_k,
                    "robustness_score_threshold": self.runtime.agent.robustness_score_threshold,
                    "max_family_budget_share": self.runtime.agent.max_family_budget_share,
                    "min_expression_novelty": self.runtime.agent.min_expression_novelty,
                    "llm_expression_enabled": self.runtime.agent.llm_expression_enabled,
                    "llm_expression_max_factors": self.runtime.agent.llm_expression_max_factors,
                    "llm_expression_max_depth": self.runtime.agent.llm_expression_max_depth,
                    "llm_expression_max_length": self.runtime.agent.llm_expression_max_length,
                    "family_filter": list(self.runtime.agent.family_filter),
                    "submission_mode": self.runtime.agent.submission_mode,
                    "allow_pending_checks": self.runtime.agent.allow_pending_checks,
                    "workdir": str(self.runtime.agent.workdir),
                },
            },
            "summary": summary,
            "leaderboard": leaderboard,
            "events": self.events,
            "hypothesis_log": self.notebook.hypothesis_log,
            "stage_history": self.notebook.stage_history,
            "submissions": [pipeline.compact_submission(item) for item in self.submission_attempts],
            "state": state,
        }
        safe_stamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.toolbox.agent_runs_dir / f"{safe_stamp}_{run_id}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path
