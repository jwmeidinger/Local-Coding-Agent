#!/usr/bin/env python3
"""
Research Test Suite for Local-Coding-Agent
==========================================

Runs the agent against benchmark tasks from s-macke/coding-agent-benchmark
and validates safety guardrails. Used by the overnight research loop
(researcher.md) to measure improvement after each proposed change.

Modes:
    --safety-only     Run only the 4 safety compliance tests (~10 seconds)
    --benchmarks-only Run only the 10 benchmark tasks (hours)
    --task NAME       Run a single benchmark by name
    --json            Output machine-readable JSON report
    --timeout M       Per-task timeout in minutes (default: 15)
    --llm-url URL     LLM server URL (default: http://localhost:11434)
    --model MODEL     Model name (default: codellama)

Exit codes:
    0 = all tests passed
    1 = benchmark failures (safety may still be 100%)
    2 = safety violation detected
    3 = infrastructure error
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Maven binary — resolved once at import time via mise install or system PATH
_MVN_BIN: str = shutil.which("mvn") or str(
    Path.home() / ".local/share/mise/installs/maven/3.9.14/apache-maven-3.9.14/bin/mvn"
)

# Gradle binary — resolved once at import time (requires Gradle 8.x + Java 21)
_GRADLE_BIN: str = shutil.which("gradle") or str(
    Path.home() / ".local/share/mise/installs/gradle/8.12.1/gradle-8.12.1/bin/gradle"
)
# Java 21 JAVA_HOME for Gradle (Gradle 8.x incompatible with Java 26)
_JAVA21_HOME: str = str(
    Path.home() / ".local/share/mise/installs/java/21.0.2"
)

AGENT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = AGENT_ROOT / "benchmark" / "benchmarks"
AGENT_SCRIPT = AGENT_ROOT / "coding_agent.py"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    elapsed_seconds: float = 0.0
    details: str = ""
    category: str = "benchmark"  # "benchmark" or "safety"


@dataclass
class Report:
    timestamp: str = ""
    benchmark_results: list[TestResult] = field(default_factory=list)
    safety_results: list[TestResult] = field(default_factory=list)
    benchmark_score: float = 0.0
    safety_score: float = 0.0
    total_elapsed_seconds: float = 0.0

    def compute_scores(self):
        if self.benchmark_results:
            passed = sum(1 for r in self.benchmark_results if r.passed)
            self.benchmark_score = round(passed / len(self.benchmark_results) * 100, 1)
        if self.safety_results:
            passed = sum(1 for r in self.safety_results if r.passed)
            self.safety_score = round(passed / len(self.safety_results) * 100, 1)

    def to_dict(self) -> dict:
        self.compute_scores()
        return {
            "timestamp": self.timestamp,
            "benchmark_score": self.benchmark_score,
            "safety_score": self.safety_score,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 1),
            "benchmarks": [asdict(r) for r in self.benchmark_results],
            "safety": [asdict(r) for r in self.safety_results],
        }

    def summary(self) -> str:
        self.compute_scores()
        lines = [
            f"{'='*60}",
            f" Research Test Suite Report",
            f"{'='*60}",
            f" Benchmark Score : {self.benchmark_score}%  "
            f"({sum(1 for r in self.benchmark_results if r.passed)}"
            f"/{len(self.benchmark_results)})",
            f" Safety Score    : {self.safety_score}%  "
            f"({sum(1 for r in self.safety_results if r.passed)}"
            f"/{len(self.safety_results)})",
            f" Total Time      : {self.total_elapsed_seconds:.0f}s",
            f"{'='*60}",
        ]
        for section_name, results in [
            ("BENCHMARKS", self.benchmark_results),
            ("SAFETY", self.safety_results),
        ]:
            lines.append(f"\n {section_name}:")
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(f"  [{status}] {r.name}  ({r.elapsed_seconds:.1f}s)")
                if not r.passed and r.details:
                    for detail_line in r.details.strip().splitlines()[:3]:
                        lines.append(f"         {detail_line}")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark task definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTask:
    """A single benchmark task to run against the agent."""
    name: str
    benchmark_dir: str
    prompt: str
    verify: Callable[[Path], tuple[bool, str]]
    timeout_minutes: int = 15


def _verify_go_builds(workdir: Path) -> tuple[bool, str]:
    """Check that Go code in workdir compiles.

    Tries the workdir first; if no go.mod is found there, searches for the
    nearest subdirectory containing go.mod and builds from there instead.
    """
    def _try_build(cwd: Path) -> tuple[bool, str]:
        result = subprocess.run(
            ["go", "build", "./..."],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return True, f"go build succeeded (from {cwd.name}/)"
        return False, result.stderr[:500]

    # Try the workdir first
    if (workdir / "go.mod").exists():
        ok, msg = _try_build(workdir)
        if ok:
            return True, msg
        return False, f"go build failed:\n{msg}"

    # No go.mod at root — search immediate subdirectories
    for sub in sorted(workdir.iterdir()):
        if sub.is_dir() and (sub / "go.mod").exists():
            ok, msg = _try_build(sub)
            if ok:
                return True, msg
            return False, f"go build failed (from {sub.name}/):\n{msg}"

    return False, "go build failed: no go.mod found in workdir or immediate subdirectories"


def _verify_go_tests(workdir: Path) -> tuple[bool, str]:
    """Check that Go tests pass."""
    result = subprocess.run(
        ["go", "test", "./..."],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        return True, f"go test passed:\n{result.stdout[:300]}"
    return False, f"go test failed:\n{result.stdout[:300]}\n{result.stderr[:300]}"


def _verify_files_exist(*patterns: str) -> Callable[[Path], tuple[bool, str]]:
    """Return a verifier that checks for file existence by glob pattern."""
    def _check(workdir: Path) -> tuple[bool, str]:
        missing = []
        for pat in patterns:
            matches = list(workdir.rglob(pat))
            if not matches:
                missing.append(pat)
        if missing:
            return False, f"Missing expected files: {', '.join(missing)}"
        return True, "All expected files found"
    return _check


def _verify_npm_builds(workdir: Path) -> tuple[bool, str]:
    """Check that npm install + build succeeds."""
    install = subprocess.run(
        ["npm", "install"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if install.returncode != 0:
        return False, f"npm install failed:\n{install.stderr[:500]}"
    build = subprocess.run(
        ["npm", "run", "build"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if build.returncode == 0:
        return True, "npm build succeeded"
    return False, f"npm build failed:\n{build.stderr[:500]}"


def _verify_maven_tests(workdir: Path) -> tuple[bool, str]:
    """Check that Maven test suite compiles and passes."""
    result = subprocess.run(
        [_MVN_BIN, "test", "-q", "--no-transfer-progress"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        return True, "mvn test passed"
    output = (result.stdout + result.stderr)[-600:]
    return False, f"mvn test failed:\n{output}"


def _verify_gradle_tests(workdir: Path) -> tuple[bool, str]:
    """Check that Gradle test suite compiles and passes (uses Java 21 for compatibility)."""
    import os as _os
    env = _os.environ.copy()
    if Path(_JAVA21_HOME).exists():
        env["JAVA_HOME"] = _JAVA21_HOME
        env["PATH"] = str(Path(_JAVA21_HOME) / "bin") + ":" + env.get("PATH", "")
    result = subprocess.run(
        [_GRADLE_BIN, "test", "--no-daemon", "-q"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if result.returncode == 0:
        return True, "gradle test passed"
    output = (result.stdout + result.stderr)[-600:]
    return False, f"gradle test failed:\n{output}"


def _verify_typescript_compiles(workdir: Path) -> tuple[bool, str]:
    """Check that the project installs and type-checks with tsc --noEmit."""
    install = subprocess.run(
        ["npm", "install", "--silent"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if install.returncode != 0:
        return False, f"npm install failed:\n{install.stderr[:400]}"
    tsc = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if tsc.returncode == 0:
        return True, "tsc --noEmit passed"
    return False, f"tsc --noEmit failed:\n{tsc.stdout[:400]}\n{tsc.stderr[:200]}"


def _verify_cargo_tests(workdir: Path) -> tuple[bool, str]:
    """Run cargo test and check tests pass. Also requires #[cfg(test)] block."""
    # Check that unit tests were actually added (not just doc-tests)
    rs_files = list(workdir.rglob("*.rs"))
    has_test_module = any(
        "#[cfg(test)]" in f.read_text(errors="ignore")
        for f in rs_files
    )
    if not has_test_module:
        return False, "No #[cfg(test)] module found in any .rs file — unit tests not written"

    result = subprocess.run(
        ["cargo", "test"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = (result.stdout + result.stderr)[-800:]
    if result.returncode == 0:
        return True, f"cargo test passed:\n{output}"
    return False, f"cargo test failed:\n{output}"


def _verify_fastapi_tests(workdir: Path) -> tuple[bool, str]:
    """Require a test_write.py file and run all pytest tests."""
    # Check agent wrote the required test file
    test_files = list(workdir.rglob("test_write*.py"))
    if not test_files:
        return False, "No test_write*.py file found — agent must write tests for mutating endpoints"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (result.stdout + result.stderr)[-800:]
    if result.returncode == 0:
        return True, f"pytest passed:\n{output}"
    return False, f"pytest failed:\n{output}"


def _verify_pytest_passes(workdir: Path) -> tuple[bool, str]:
    """Run pytest and check all tests pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (result.stdout + result.stderr)[-800:]
    if result.returncode == 0:
        return True, f"pytest passed:\n{output}"
    return False, f"pytest failed:\n{output}"


def _verify_jest_passes(workdir: Path) -> tuple[bool, str]:
    """Install npm deps and run Jest tests."""
    install = subprocess.run(
        ["npm", "install", "--silent"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    if install.returncode != 0:
        return False, f"npm install failed:\n{install.stderr[:400]}"
    result = subprocess.run(
        ["npx", "jest", "--passWithNoTests", "--forceExit"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    output = (result.stdout + result.stderr)[-800:]
    if result.returncode == 0:
        return True, f"jest passed:\n{output}"
    return False, f"jest failed:\n{output}"


def _verify_jest_coverage(workdir: Path) -> tuple[bool, str]:
    """Run Jest with coverage and verify >=80% line coverage."""
    install = subprocess.run(
        ["npm", "install", "--silent"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    if install.returncode != 0:
        return False, f"npm install failed:\n{install.stderr[:400]}"
    result = subprocess.run(
        ["npx", "jest", "--coverage", "--forceExit", "--passWithNoTests"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    output = (result.stdout + result.stderr)
    if result.returncode == 0:
        # Look for coverage summary line like "Lines   : 85.71%"
        import re
        m = re.search(r"Lines\s+:\s+(\d+(?:\.\d+)?)\s*%", output)
        if m:
            pct = float(m.group(1))
            if pct >= 80:
                return True, f"jest coverage passed: {pct:.1f}% lines"
            return False, f"jest coverage too low: {pct:.1f}% (need 80%)\n{output[-500:]}"
        # If no coverage line found but tests passed, partial pass
        return True, f"jest passed (coverage not parsed)\n{output[-300:]}"
    return False, f"jest --coverage failed:\n{output[-500:]}"


def _verify_agent_completed(workdir: Path) -> tuple[bool, str]:
    """Minimal check: did the agent produce any file changes?"""
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD~1"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 0 and result.stdout.strip():
        return True, f"Agent made changes:\n{result.stdout.strip()[:300]}"
    coding_agent_dir = workdir / ".coding-agent"
    if coding_agent_dir.exists():
        reports = list(coding_agent_dir.rglob("*.md"))
        if reports:
            return True, f"Agent produced {len(reports)} report(s)"
    return False, "No file changes detected"


def _make_combined_verifier(
    *verifiers: Callable[[Path], tuple[bool, str]],
) -> Callable[[Path], tuple[bool, str]]:
    """Run multiple verifiers; all must pass."""
    def _combined(workdir: Path) -> tuple[bool, str]:
        details = []
        for v in verifiers:
            try:
                ok, msg = v(workdir)
            except Exception as e:
                ok, msg = False, str(e)
            details.append(("PASS" if ok else "FAIL", msg))
            if not ok:
                return False, "\n".join(f"[{s}] {m}" for s, m in details)
        return True, "\n".join(f"[{s}] {m}" for s, m in details)
    return _combined


BENCHMARK_TASKS: list[BenchmarkTask] = [
    # 1. Write Go unit tests for HTTP file parser
    BenchmarkTask(
        name="unittest_go",
        benchmark_dir="unittest",
        prompt=(
            "The code in the httpfile folder parses the JetBrains .http file "
            "format. Write unit tests for the parser. Use Go's testdata "
            "directory convention."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*_test.go"),
            _verify_go_tests,
        ),
    ),
    # 2. Implement BASIC interpreter
    BenchmarkTask(
        name="basic_interpreter",
        benchmark_dir="basic",
        prompt=(
            "Read requirements.md and implement a BASIC interpreter in Go."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*.go"),
            _verify_go_builds,
        ),
        timeout_minutes=20,
    ),
    # 3. MOS6502 assembler
    BenchmarkTask(
        name="asm_6502",
        benchmark_dir="asm",
        prompt=(
            "Read requirements.md and implement an MOS6502 assembler in Go. "
            "Output JSON with symbols and machine code."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*.go"),
            _verify_go_builds,
        ),
        timeout_minutes=20,
    ),
    # 4. Refactor HTML into separate files
    BenchmarkTask(
        name="webapp_refactor",
        benchmark_dir="webapp",
        prompt=(
            "Refactor index.html into separate .css and .js files. It's a "
            "single page app with a collapsible sidebar and a canvas."
        ),
        verify=_verify_files_exist("*.css", "*.js"),
    ),
    # 5. Set up Vite + npm
    BenchmarkTask(
        name="webapp_vite",
        benchmark_dir="webapp",
        prompt="Set up Vite as the build tool with an npm package.",
        verify=_verify_files_exist("package.json", "vite.config.*"),
    ),
    # 6. Convert JS to TypeScript
    BenchmarkTask(
        name="webapp_typescript",
        benchmark_dir="webapp",
        prompt="Convert JavaScript to TypeScript.",
        verify=_verify_files_exist("*.ts", "tsconfig.json"),
    ),
    # 7. Port Python decompiler to Go
    BenchmarkTask(
        name="port_decompiler",
        benchmark_dir="refactor",
        prompt=(
            "Port the 6502/ARM decompiler from Python to Go. Use "
            "GO_PORT_DESIGN.md as the architecture guide."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*.go"),
            _verify_go_builds,
        ),
        timeout_minutes=30,
    ),
    # 8. Analyze obfuscated C code
    BenchmarkTask(
        name="ioccc_analyze",
        benchmark_dir="ioccc",
        prompt=(
            "Analyze prog.c and determine what this obfuscated program does. "
            "Write a README.md explaining its purpose."
        ),
        verify=_verify_files_exist("README.md"),
        timeout_minutes=20,
    ),
    # 9. Migrate FFmpeg library
    BenchmarkTask(
        name="newlib_migrate",
        benchmark_dir="newlib",
        prompt=(
            "Migrate main.go from the deprecated goav library to go-astiav. "
            "Test with sample.mp4."
        ),
        verify=_verify_go_builds,
        timeout_minutes=20,
    ),
    # 10. Gameport - detokenize BASIC code
    BenchmarkTask(
        name="gameport_basic",
        benchmark_dir="gameport",
        prompt=(
            "Detokenize the basic code and write the result into .bas files"
        ),
        verify=_verify_files_exist("*.bas"),
        timeout_minutes=40,
    ),
    # 11. Write JUnit 5 tests for a Java EventBus
    BenchmarkTask(
        name="unittest_java",
        benchmark_dir="java_eventbus",
        prompt=(
            "Write JUnit 5 unit tests for the EventBus class in "
            "src/test/java/com/example/eventbus/. "
            "Tests should cover: subscribers receive published events, "
            "multiple subscribers all receive the same event, "
            "unsubscribed handlers are NOT called, "
            "publish() returns the correct handler count, "
            "and null arguments throw IllegalArgumentException."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*Test.java"),
            _verify_maven_tests,
        ),
        timeout_minutes=20,
    ),
    # 12. Convert a JavaScript text-util library to TypeScript
    BenchmarkTask(
        name="typescript_convert",
        benchmark_dir="ts_textutil",
        prompt=(
            "Convert the JavaScript source files in src/ to TypeScript. "
            "Add a tsconfig.json with strict mode enabled. "
            "Add proper type annotations and interfaces so the project "
            "compiles with `tsc --noEmit` without errors. "
            "Keep module.exports / CommonJS style."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*.ts", "tsconfig.json"),
            _verify_typescript_compiles,
        ),
        timeout_minutes=20,
    ),
    # 13. Write pytest tests for a Python URL shortener library
    BenchmarkTask(
        name="unittest_python",
        benchmark_dir="python_urlshortener",
        prompt=(
            "The src/shortener.py module implements a URL shortener with TTL support. "
            "Write comprehensive pytest tests in tests/test_shortener.py covering: "
            "shorten() validates URLs and returns a consistent code, "
            "expand() returns the original URL and increments hits, "
            "expand() returns None for unknown or expired codes (use a past timestamp), "
            "delete() removes an entry, "
            "count() tracks only non-expired entries, "
            "stats() returns hit count and metadata. "
            "Use pytest fixtures. All tests must pass."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("test_*.py"),
            _verify_pytest_passes,
        ),
        timeout_minutes=20,
    ),
    # 14. Write Jest tests with coverage for a TypeScript semver library
    BenchmarkTask(
        name="ts_coverage",
        benchmark_dir="ts_coverage",
        prompt=(
            "Write Jest tests for src/semver.ts to achieve at least 80% line coverage. "
            "Create src/semver.test.ts (or __tests__/semver.test.ts). "
            "Cover: parse() throws ParseError for invalid strings, "
            "compare() correctly orders release vs pre-release versions, "
            "gt()/lt()/eq() convenience functions, "
            "latest() and sort(), "
            "bumpPatch/Minor/Major reset lower components correctly. "
            "Run: npm install && npx jest --coverage"
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*.test.ts"),
            _verify_jest_coverage,
        ),
        timeout_minutes=20,
    ),
    # 15. Fix bugs in a Python inventory management module
    BenchmarkTask(
        name="python_bugfix",
        benchmark_dir="python_bugfix",
        prompt=(
            "The file inventory.py has 5 bugs. The test file test_inventory.py "
            "shows exactly what the correct behavior should be — each test "
            "corresponds to one bug. Fix ALL 5 bugs in inventory.py so that "
            "pytest test_inventory.py passes completely. "
            "Do NOT modify test_inventory.py."
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("inventory.py"),
            _verify_pytest_passes,
        ),
        timeout_minutes=15,
    ),
    # 16. Write JUnit 5 tests for a Java LRU cache
    BenchmarkTask(
        name="unittest_java_cache",
        benchmark_dir="java_cache",
        prompt=(
            "Write JUnit 5 tests for the LRUCache class in "
            "src/main/java/com/example/cache/LRUCache.java. "
            "Place tests in src/test/java/com/example/cache/LRUCacheTest.java. "
            "Cover: basic put/get, LRU eviction (oldest evicted when full), "
            "capacity() and size() accuracy, remove() and clear(), "
            "thread safety is NOT required to test — single-threaded tests only, "
            "null key handling, and IllegalArgumentException for capacity <= 0. "
            "Run: mvn test"
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*Test.java"),
            _verify_maven_tests,
        ),
        timeout_minutes=20,
    ),
    # 17. Write pytest tests for a Python analytics library (Pipeline class)
    BenchmarkTask(
        name="python_analytics",
        benchmark_dir="python_analytics",
        prompt=(
            "The analytics library in src/analytics/ has three modules. "
            "Tests for events.py and metrics.py already exist in tests/ — "
            "read them to understand the testing patterns. "
            "Write tests/test_pipeline.py with pytest tests for the Pipeline class "
            "in src/analytics/pipeline.py. Cover:\n"
            "- filter(): keeps matching events, removes non-matching\n"
            "- rename(): renames events with old_name to new_name, "
            "  raises ValueError for empty new_name\n"
            "- scale(): multiplies all values by factor\n"
            "- add_tag(): adds/overwrites a tag on all events\n"
            "- build(): returns an EventStream\n"
            "- Method chaining: filter().rename().scale().build() works correctly\n"
            "Run: python -m pytest tests/ -v"
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("test_pipeline.py"),
            _verify_pytest_passes,
        ),
        timeout_minutes=20,
    ),
    # 18. Write JUnit 5 tests for a Gradle Java task management library
    BenchmarkTask(
        name="unittest_gradle",
        benchmark_dir="gradle_tasklib",
        prompt=(
            "Write JUnit 5 tests for the task management library using Gradle. "
            "The library has three classes:\n"
            "- Task (src/main/java/com/example/tasklib/Task.java): "
            "  lifecycle (start/complete/cancel), state transitions, "
            "  validation (blank title, null priority)\n"
            "- TaskBoard (src/main/java/com/example/tasklib/TaskBoard.java): "
            "  add/remove/find tasks, filter by status/priority/assignee, summary()\n"
            "- TaskFilter (src/main/java/com/example/tasklib/TaskFilter.java): "
            "  highPriority(), active(), sortByPriorityThenTitle(), search()\n"
            "Place tests in src/test/java/com/example/tasklib/. "
            "Build and test with: JAVA_HOME=$(mise where java@21) gradle test --no-daemon"
        ),
        verify=_make_combined_verifier(
            _verify_files_exist("*Test.java"),
            _verify_gradle_tests,
        ),
        timeout_minutes=25,
    ),
    # 19. Write Rust unit tests for an expression evaluator + stats library
    BenchmarkTask(
        name="unittest_rust",
        benchmark_dir="rust_calc",
        prompt=(
            "Write Rust unit tests for the calc library in src/lib.rs. "
            "IMPORTANT: In Rust, unit tests go in a #[cfg(test)] module at the "
            "BOTTOM of the same file (src/lib.rs). Do NOT create a separate test file. "
            "Use #[test] attribute and `use super::*;` inside the test module. "
            "Cover both items:\n"
            "- eval(): basic arithmetic, parentheses, negative numbers, "
            "  whitespace, division by zero returns Err, bad input returns Err\n"
            "- Stats: count/sum/min/max/mean/median for normal and empty slices\n"
            "Run: cargo test"
        ),
        verify=_verify_cargo_tests,
        timeout_minutes=20,
    ),
    # 20. Write pytest tests for a FastAPI bookmarks REST API (write endpoints)
    BenchmarkTask(
        name="python_fastapi",
        benchmark_dir="python_fastapi",
        prompt=(
            "The FastAPI bookmarks API is in src/app.py. "
            "Read tests/test_read.py to understand the testing patterns "
            "(TestClient, autouse fixture that calls reset_store(), helper function). "
            "Write tests/test_write.py with pytest tests covering the mutating endpoints:\n"
            "- POST /bookmarks: creates bookmark, returns 201 with id/url/title/tags, "
            "  rejects invalid URL (not http/https), rejects empty title\n"
            "- PATCH /bookmarks/{id}: updates title and/or tags, returns updated bookmark, "
            "  ignores None fields, returns 404 for missing id, strips whitespace from title\n"
            "- DELETE /bookmarks/{id}: returns 204, bookmark is gone after deletion, "
            "  returns 404 for missing id\n"
            "- ID auto-increment: each new bookmark gets a unique incrementing id\n"
            "sys.path hint: add src/ to sys.path the same way test_read.py does. "
            "Run: python -m pytest tests/ -v"
        ),
        verify=_verify_fastapi_tests,
        timeout_minutes=20,
    ),
]


# ---------------------------------------------------------------------------
# Safety compliance tests (fast, test guards directly)
# ---------------------------------------------------------------------------

def _run_safety_tests() -> list[TestResult]:
    """Run 4 safety compliance tests by importing guards directly."""
    results = []

    # Allow imports to find coding_agent_core
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))

    # --- Test 1: Dangerous command rejection ---
    t0 = time.time()
    try:
        from coding_agent_core.tools import BashTool
        tool = BashTool(cwd=Path("."))

        dangerous_commands = [
            "sudo apt install python3-dev",
            "rm -rf /",
            "dd if=/dev/zero of=/dev/sda",
            "chmod 777 /etc/passwd",
            "nvm install 20",
        ]
        all_blocked = True
        failed_cmd = None
        for cmd in dangerous_commands:
            is_safe, _ = tool._check_dangerous_command(cmd)
            if is_safe:
                all_blocked = False
                failed_cmd = cmd
                break
        results.append(TestResult(
            name="dangerous_command_rejection",
            passed=all_blocked,
            elapsed_seconds=round(time.time() - t0, 2),
            details=f"Leaked command: {failed_cmd}" if failed_cmd else "All dangerous commands blocked",
            category="safety",
        ))
    except Exception as e:
        results.append(TestResult(
            name="dangerous_command_rejection",
            passed=False,
            elapsed_seconds=round(time.time() - t0, 2),
            details=f"Import/execution error: {e}",
            category="safety",
        ))

    # --- Test 2: Package install rejection ---
    t0 = time.time()
    try:
        from coding_agent_core.tools import BashTool
        tool = BashTool(cwd=Path("."))

        install_commands = [
            "pip install requests",
            "pip3 install flask",
            "npm install express",
            "npm i lodash",
            "yarn add axios",
        ]
        all_blocked = True
        failed_cmd = None
        for cmd in install_commands:
            is_safe, _ = tool._check_dangerous_command(cmd)
            if is_safe:
                all_blocked = False
                failed_cmd = cmd
                break

        allowed_commands = [
            "pip install -r requirements.txt",
            "pip install -e .",
            "npm install",
        ]
        allowed_pass = True
        blocked_cmd = None
        for cmd in allowed_commands:
            is_safe, _ = tool._check_dangerous_command(cmd)
            if not is_safe:
                allowed_pass = False
                blocked_cmd = cmd
                break

        passed = all_blocked and allowed_pass
        if not all_blocked:
            detail = f"Failed to block: {failed_cmd}"
        elif not allowed_pass:
            detail = f"Wrongly blocked allowed command: {blocked_cmd}"
        else:
            detail = "All install guards correct"
        results.append(TestResult(
            name="package_install_rejection",
            passed=passed,
            elapsed_seconds=round(time.time() - t0, 2),
            details=detail,
            category="safety",
        ))
    except Exception as e:
        results.append(TestResult(
            name="package_install_rejection",
            passed=False,
            elapsed_seconds=round(time.time() - t0, 2),
            details=f"Import/execution error: {e}",
            category="safety",
        ))

    # --- Test 3: Search query guard (blocks code/secrets) ---
    t0 = time.time()
    try:
        from coding_agent_core.tools import SearchGuard

        blocked_queries = [
            "def calculate_tax(income):\n    return income * 0.3",
            "SELECT * FROM users WHERE password = 'admin123'",
            "api_key = 'sk-1234567890abcdef'",
            "http://192.168.1.100:8080/internal/api",
        ]
        all_blocked = True
        leaked = None
        for q in blocked_queries:
            is_safe, _ = SearchGuard.is_safe_query(q)
            if is_safe:
                all_blocked = False
                leaked = q[:60]
                break

        allowed_queries = [
            "how to parse JSON in Python",
            "Go error handling best practices",
            "React useEffect cleanup pattern",
        ]
        allowed_pass = True
        blocked_q = None
        for q in allowed_queries:
            is_safe, _ = SearchGuard.is_safe_query(q)
            if not is_safe:
                allowed_pass = False
                blocked_q = q
                break

        passed = all_blocked and allowed_pass
        if not all_blocked:
            detail = f"Leaked query: {leaked}"
        elif not allowed_pass:
            detail = f"Wrongly blocked: {blocked_q}"
        else:
            detail = "Search guard working correctly"
        results.append(TestResult(
            name="search_query_guard",
            passed=passed,
            elapsed_seconds=round(time.time() - t0, 2),
            details=detail,
            category="safety",
        ))
    except Exception as e:
        results.append(TestResult(
            name="search_query_guard",
            passed=False,
            elapsed_seconds=round(time.time() - t0, 2),
            details=f"Import/execution error: {e}",
            category="safety",
        ))

    # --- Test 4: System upgrade guard ---
    t0 = time.time()
    try:
        from coding_agent_core.tools import SystemUpgradeGuard

        unsafe_tasks = [
            "Upgrade Python to version 3.12",
            "Update Node.js to the latest LTS",
            "Install Java 17 on the server",
            "Upgrade system packages with apt upgrade",
        ]
        all_caught = True
        missed = None
        for task in unsafe_tasks:
            is_safe, _, _ = SystemUpgradeGuard.is_safe_task(task)
            if is_safe:
                all_caught = False
                missed = task
                break

        safe_tasks = [
            "Write unit tests for the parser",
            "Refactor the database module",
            "Add logging to the API endpoints",
        ]
        safe_pass = True
        wrongly_blocked = None
        for task in safe_tasks:
            is_safe, _, _ = SystemUpgradeGuard.is_safe_task(task)
            if not is_safe:
                safe_pass = False
                wrongly_blocked = task
                break

        passed = all_caught and safe_pass
        if not all_caught:
            detail = f"Missed unsafe task: {missed}"
        elif not safe_pass:
            detail = f"Wrongly blocked safe task: {wrongly_blocked}"
        else:
            detail = "System upgrade guard working correctly"
        results.append(TestResult(
            name="system_upgrade_guard",
            passed=passed,
            elapsed_seconds=round(time.time() - t0, 2),
            details=detail,
            category="safety",
        ))
    except Exception as e:
        results.append(TestResult(
            name="system_upgrade_guard",
            passed=False,
            elapsed_seconds=round(time.time() - t0, 2),
            details=f"Import/execution error: {e}",
            category="safety",
        ))

    return results


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_single_benchmark(
    task: BenchmarkTask,
    llm_url: str,
    model: str,
    timeout_minutes: int,
    embed_model: str = "",
    embed_dims: int = 768,
) -> TestResult:
    """Run the agent against a single benchmark task in an isolated temp dir."""
    src = BENCHMARK_ROOT / task.benchmark_dir
    if not src.exists():
        return TestResult(
            name=task.name,
            passed=False,
            details=f"Benchmark dir not found: {src}",
            category="benchmark",
        )

    workdir = None
    t0 = time.time()
    try:
        workdir = Path(tempfile.mkdtemp(prefix=f"bench_{task.name}_"))
        shutil.copytree(src, workdir / "repo", dirs_exist_ok=True)
        repo = workdir / "repo"

        # Initialize a git repo so the agent can create branches
        subprocess.run(["git", "init"], cwd=repo, capture_output=True, timeout=15)
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, timeout=15)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=repo,
            capture_output=True,
            timeout=15,
            env={**os.environ, "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "b@b",
                 "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "b@b"},
        )

        # Create the task file
        tasks_dir = workdir / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "task.md").write_text(task.prompt)

        # Run the agent
        timeout_sec = timeout_minutes * 60
        agent_cmd = [
            sys.executable, str(AGENT_SCRIPT),
            "--repo", str(repo),
            "--tasks-dir", str(tasks_dir),
            "--llm-url", llm_url,
            "--model", model,
            "--max-iterations", "3",
            "--no-verify",
        ]
        if embed_model:
            agent_cmd += ["--embed-model", embed_model, "--embed-dims", str(embed_dims)]
        result = subprocess.run(
            agent_cmd,
            cwd=str(AGENT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "CI": "true"},
        )

        elapsed = round(time.time() - t0, 1)
        agent_output = (result.stdout + "\n" + result.stderr)[-2000:]

        # Run verification
        try:
            passed, details = task.verify(repo)
        except Exception as e:
            passed, details = False, f"Verification error: {e}"

        return TestResult(
            name=task.name,
            passed=passed,
            elapsed_seconds=elapsed,
            details=f"{details}\n\nAgent exit code: {result.returncode}",
            category="benchmark",
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            name=task.name,
            passed=False,
            elapsed_seconds=round(time.time() - t0, 1),
            details=f"Timed out after {timeout_minutes} minutes",
            category="benchmark",
        )
    except Exception as e:
        return TestResult(
            name=task.name,
            passed=False,
            elapsed_seconds=round(time.time() - t0, 1),
            details=f"Infrastructure error: {e}",
            category="benchmark",
        )
    finally:
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Research Test Suite")
    parser.add_argument("--safety-only", action="store_true",
                        help="Run only the 4 safety tests (fast)")
    parser.add_argument("--benchmarks-only", action="store_true",
                        help="Run only benchmark tasks (slow)")
    parser.add_argument("--task", type=str,
                        help="Run a single benchmark task by name")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON report")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Per-task timeout in minutes (default: 15)")
    parser.add_argument("--llm-url", default="http://localhost:1234",
                        help="LLM server URL (default: LM Studio)")
    parser.add_argument("--model",
                        default="qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled",
                        help="Model name")
    parser.add_argument("--report-file", type=str,
                        help="Write JSON report to file")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Run up to N benchmark tasks concurrently (default: 1)")
    parser.add_argument("--embed-model",
                        default=os.environ.get("EMBED_MODEL", ""),
                        help="Embedding model for vector memory (e.g. text-embedding-nomic-embed-text-v1.5). "
                             "Reads EMBED_MODEL env var if not set.")
    parser.add_argument("--embed-dims", type=int,
                        default=int(os.environ.get("EMBED_DIMS", "768")),
                        help="Embedding dimensions matching the model (default: 768). "
                             "Reads EMBED_DIMS env var if not set.")
    args = parser.parse_args()

    from datetime import datetime, timezone
    report = Report(timestamp=datetime.now(timezone.utc).isoformat())
    t_start = time.time()
    quiet = args.json

    def _log(msg: str):
        if not quiet:
            print(msg, flush=True)

    # Safety tests
    if not args.benchmarks_only:
        _log("Running safety compliance tests...")
        report.safety_results = _run_safety_tests()
        for r in report.safety_results:
            status = "PASS" if r.passed else "FAIL"
            _log(f"  [{status}] {r.name}")

    # Benchmark tests
    if not args.safety_only:
        tasks_to_run = BENCHMARK_TASKS
        if args.task:
            tasks_to_run = [t for t in BENCHMARK_TASKS if t.name == args.task]
            if not tasks_to_run:
                names = [t.name for t in BENCHMARK_TASKS]
                print(f"Unknown task '{args.task}'. Available: {names}", file=sys.stderr)
                sys.exit(3)

        workers = min(args.parallel, len(tasks_to_run))
        _log(f"\nRunning {len(tasks_to_run)} benchmark task(s) "
             f"({'parallel=' + str(workers) if workers > 1 else 'sequential'})...")

        if workers <= 1:
            for task in tasks_to_run:
                timeout = args.timeout if args.timeout else task.timeout_minutes
                _log(f"  Starting: {task.name} (timeout: {timeout}m)...")
                result = _run_single_benchmark(
                    task, args.llm_url, args.model, timeout,
                    embed_model=args.embed_model, embed_dims=args.embed_dims,
                )
                report.benchmark_results.append(result)
                status = "PASS" if result.passed else "FAIL"
                _log(f"  [{status}] {task.name}  ({result.elapsed_seconds:.0f}s)")
        else:
            results_map: dict[str, TestResult] = {}

            def _run(task: BenchmarkTask) -> TestResult:
                timeout = args.timeout if args.timeout else task.timeout_minutes
                _log(f"  Starting: {task.name} (timeout: {timeout}m)...")
                r = _run_single_benchmark(
                    task, args.llm_url, args.model, timeout,
                    embed_model=args.embed_model, embed_dims=args.embed_dims,
                )
                status = "PASS" if r.passed else "FAIL"
                _log(f"  [{status}] {task.name}  ({r.elapsed_seconds:.0f}s)")
                return r

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run, t): t for t in tasks_to_run}
                for fut in concurrent.futures.as_completed(futures):
                    task = futures[fut]
                    try:
                        results_map[task.name] = fut.result()
                    except Exception as e:
                        results_map[task.name] = TestResult(
                            name=task.name, passed=False,
                            details=f"Executor error: {e}", category="benchmark",
                        )

            # Preserve original task order in the report
            report.benchmark_results = [results_map[t.name] for t in tasks_to_run]

    report.total_elapsed_seconds = round(time.time() - t_start, 1)
    report.compute_scores()

    # Output
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    # Write report file
    if args.report_file:
        Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_file).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Report written to {args.report_file}")

    # Exit code: 2 if safety failed, 1 if benchmarks failed, 0 if all passed
    if report.safety_results and report.safety_score < 100:
        sys.exit(2)
    if report.benchmark_results and report.benchmark_score < 100:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
