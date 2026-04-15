#!/usr/bin/env python3
"""Run historically stable benchmark tasks as a fast end-of-session check.

This wrapper does NOT modify tests/research_test_suite.py (the objective harness).
It simply invokes that harness with --task for a curated stable subset and prints
an aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STABLE_TASKS: list[str] = [
    "ioccc_analyze",
    "python_bugfix",
    "python_fastapi",
    "unittest_java",
    "unittest_rust",
    "webapp_refactor",
]


def run_task(
    python_exe: str,
    suite_script: Path,
    task: str,
    llm_url: str,
    model: str,
    timeout: int,
) -> dict:
    with tempfile.NamedTemporaryFile(prefix=f"stable_{task}_", suffix=".json", delete=False) as tmp:
        report_file = tmp.name
    cmd = [
        python_exe,
        str(suite_script),
        "--task",
        task,
        "--json",
        "--llm-url",
        llm_url,
        "--model",
        model,
        "--timeout",
        str(timeout),
        "--report-file",
        report_file,
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - started, 1)

    result = {
        "task": task,
        "return_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "passed": False,
        "details": "",
        "raw_stdout": proc.stdout[-1500:],
        "raw_stderr": proc.stderr[-1500:],
    }

    try:
        report = json.loads(Path(report_file).read_text())
        bench = report.get("benchmarks", [])
        if bench:
            result["passed"] = bool(bench[0].get("passed", False))
            result["details"] = str(bench[0].get("details", ""))[:1000]
    except Exception:
        # Keep subprocess output in the result for debugging.
        pass
    finally:
        try:
            Path(report_file).unlink(missing_ok=True)
        except Exception:
            pass

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stable benchmark subset via tests/research_test_suite.py"
    )
    parser.add_argument(
        "--python",
        default=str(Path(__file__).resolve().parent / ".venv" / "bin" / "python"),
        help="Python executable to run benchmark suite (default: project .venv/bin/python)",
    )
    parser.add_argument(
        "--suite-script",
        default=str(Path(__file__).resolve().parent / "tests" / "research_test_suite.py"),
        help="Path to research_test_suite.py",
    )
    parser.add_argument("--llm-url", default="http://localhost:1234", help="LLM server URL")
    parser.add_argument("--model", default="qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled")
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Per-task timeout in minutes (default: 45)",
    )
    parser.add_argument(
        "--report-file",
        default="",
        help="Optional path to write aggregate JSON report",
    )
    args = parser.parse_args()

    suite_script = Path(args.suite_script)
    if not suite_script.exists():
        print(f"Suite script not found: {suite_script}", file=sys.stderr)
        return 3

    print(f"Running stable subset ({len(STABLE_TASKS)} tasks)...")
    print(", ".join(STABLE_TASKS))
    print()

    all_results: list[dict] = []
    started = time.time()
    for task in STABLE_TASKS:
        result = run_task(
            python_exe=args.python,
            suite_script=suite_script,
            task=task,
            llm_url=args.llm_url,
            model=args.model,
            timeout=args.timeout,
        )
        all_results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {task} ({result['elapsed_seconds']:.0f}s)")
        if not result["passed"]:
            detail = result["details"] or result["raw_stderr"] or result["raw_stdout"]
            print(f"  -> {detail.splitlines()[0] if detail else 'no details'}")

    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    score = round((passed / total) * 100, 1) if total else 0.0
    total_elapsed = round(time.time() - started, 1)

    aggregate = {
        "benchmark_score": score,
        "passed": passed,
        "total": total,
        "total_elapsed_seconds": total_elapsed,
        "tasks": all_results,
        "stable_tasks": STABLE_TASKS,
    }

    print()
    print(f"Stable subset score: {score}% ({passed}/{total})")
    print(f"Total time: {total_elapsed:.0f}s")

    if args.report_file:
        out = Path(args.report_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(aggregate, indent=2))
        print(f"Wrote report: {out}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
