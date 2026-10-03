#!/usr/bin/env python3
"""Print readable alpha history from the agent JSONL result store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def load_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid = 0
    if not path.exists():
        raise FileNotFoundError(path)

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            invalid += 1
    return records, invalid


def number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def status(record: dict[str, Any]) -> str:
    if record.get("precheck_submit_ready") is True:
        return "SUBMIT-READY"
    if record.get("quality_checks_ready") is True:
        failed = record.get("failed_correlation_checks") or []
        if failed:
            return "QUALITY-READY / CORRELATION BLOCKED"
        return "QUALITY-READY / NOT SUBMIT-READY"
    return "NOT READY"


def failed_checks(record: dict[str, Any]) -> str:
    failed = record.get("failed_checks") or []
    pending = record.get("pending_checks") or []
    parts: list[str] = []
    if failed:
        parts.append("failed=" + ", ".join(str(item) for item in failed))
    if pending:
        parts.append("pending=" + ", ".join(str(item) for item in pending))
    return "; ".join(parts) or "none"


def sort_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: float(item.get("score")) if item.get("score") is not None else float("-inf"),
        reverse=True,
    )


def print_record(rank: int, record: dict[str, Any]) -> None:
    metrics = record.get("metrics") or {}
    settings = record.get("settings") or {}
    print(f"[{rank}] {record.get('alpha_id') or '(no alpha id)'}")
    print(f"    status:     {status(record)}")
    print(f"    score:      {number(record.get('score'))}")
    print(f"    family:     {record.get('family') or '-'}")
    print(f"    evaluated:  {record.get('evaluated_at') or '-'}")
    print(
        "    simulation: "
        f"region={settings.get('region', '-')}  "
        f"universe={settings.get('universe', '-')}  "
        f"delay={settings.get('delay', '-')}  "
        f"decay={settings.get('decay', '-')}  "
        f"neutralization={settings.get('neutralization', '-')}  "
        f"truncation={settings.get('truncation', '-')}"
    )
    print(
        "    options:    "
        f"pasteurization={settings.get('pasteurization', '-')}  "
        f"unitHandling={settings.get('unitHandling', '-')}  "
        f"nanHandling={settings.get('nanHandling', '-')}  "
        f"language={settings.get('language', '-')}"
    )
    print(
        "    metrics:    "
        f"Sharpe={number(metrics.get('sharpe'))}  "
        f"Fitness={number(metrics.get('fitness'))}  "
        f"Turnover={number(metrics.get('turnover'))}  "
        f"Returns={number(metrics.get('returns'))}  "
        f"Drawdown={number(metrics.get('drawdown'))}"
    )
    print(f"    checks:     {failed_checks(record)}")
    print(f"    expression: {record.get('expression') or '-'}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print readable alpha history.")
    parser.add_argument(
        "path",
        nargs="?",
        default=".alpha_agent/results.jsonl",
        help="Path to results.jsonl (default: .alpha_agent/results.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum records to print (default: 10)")
    parser.add_argument("--only-quality-ready", action="store_true")
    parser.add_argument("--only-submit-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.only_quality_ready and args.only_submit_ready:
        print("Choose only one of --only-quality-ready and --only-submit-ready.", file=sys.stderr)
        return 2
    if args.limit < 1:
        print("--limit must be at least 1.", file=sys.stderr)
        return 2

    try:
        records, invalid = load_records(Path(args.path))
    except FileNotFoundError:
        print(f"Results file not found: {args.path}", file=sys.stderr)
        return 1

    if args.only_submit_ready:
        records = [item for item in records if item.get("precheck_submit_ready") is True]
    elif args.only_quality_ready:
        records = [item for item in records if item.get("quality_checks_ready") is True]

    records = sort_records(records)[: args.limit]
    print(f"Alpha history: {len(records)} record(s) shown")
    if invalid:
        print(f"Warning: skipped {invalid} invalid JSON line(s)")
    print("Sorted by score, highest first.\n")

    if not records:
        print("No matching alpha records.")
        return 0
    for rank, record in enumerate(records, 1):
        print_record(rank, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
