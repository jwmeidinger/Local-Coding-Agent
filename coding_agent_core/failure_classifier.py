"""Rules-based failure classifier for the coding agent.

Classifies tool/build/test output into actionable failure types so the
retry logic can choose a targeted recovery strategy instead of blindly
retrying.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Failure types
# ---------------------------------------------------------------------------

SYNTAX_ERROR = "syntax_error"
TEST_FAILURE = "test_failure"
LINT_FAILURE = "lint_failure"
TYPECHECK_FAILURE = "typecheck_failure"
MISSING_DEPENDENCY = "missing_dependency"
REVIEW_REJECTION = "review_rejection"
UNKNOWN_FAILURE = "unknown_failure"


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class FailureInfo:
    """Structured description of a failure."""

    failure_type: str
    summary: str  # One-liner for storage / display
    raw_output: str = ""  # First ~500 chars of the original output
    file_hint: Optional[str] = None  # File most likely responsible
    line_hint: Optional[int] = None  # Line number if available

    # Recommended retry budget for this failure type
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

_SYNTAX_PATTERNS: list[re.Pattern] = [
    re.compile(r"SyntaxError:", re.IGNORECASE),
    re.compile(r"IndentationError:", re.IGNORECASE),
    re.compile(r"Unexpected token", re.IGNORECASE),
    re.compile(r"Parsing error:", re.IGNORECASE),
    # Go compiler errors
    re.compile(r"^#\s+\S+\s*$", re.MULTILINE),          # "# package/path" header in go build output
    re.compile(r"\bundefined:\s+\w+", re.IGNORECASE),    # "undefined: someVar"
    re.compile(r"expected statement, found"),              # invalid placeholder like "..."
    re.compile(r"syntax error: unexpected"),               # generic Go syntax error
    re.compile(r"imported and not used:", re.IGNORECASE), # Go: unused import
    re.compile(r"declared and not used:", re.IGNORECASE), # Go: unused variable
    re.compile(r"cannot use .+ as .+type", re.IGNORECASE),   # Go: type mismatch
    re.compile(r"cannot convert .+ to type", re.IGNORECASE), # Go: type conversion error
    re.compile(r"missing return at end of function"),         # Go: missing return
    re.compile(r"not enough (arguments|return values)"),      # Go: wrong arg count
    re.compile(r"too many (arguments|return values)"),        # Go: wrong arg count
    re.compile(r"cannot index\b", re.IGNORECASE),             # Go: indexing a non-slice/map
    re.compile(r"has no field or method\b", re.IGNORECASE),  # Go: wrong field/method name
    re.compile(r"cannot take the address of", re.IGNORECASE), # Go: addressability error
    re.compile(r"invalid operation:.*\(", re.IGNORECASE),     # Go: generic invalid op
    re.compile(r"multiple-value .* used in single-value context", re.IGNORECASE),
    re.compile(r"non-name .* on left side of :=", re.IGNORECASE),
    re.compile(r"invalid append:", re.IGNORECASE),              # Go: append to non-slice (e.g. *[]T)
    re.compile(r"argument must be a slice", re.IGNORECASE),     # Go: append wrong type
    re.compile(r"cannot assign to .+\(.*not addressable\)", re.IGNORECASE),  # Go: non-addressable
    re.compile(r"must be a slice type", re.IGNORECASE),         # Go: range/append on pointer
]

_TEST_PATTERNS: list[re.Pattern] = [
    re.compile(r"FAIL\s+(src/|tests/|test/)", re.IGNORECASE),
    re.compile(r"FAILED\s+tests?/", re.IGNORECASE),
    re.compile(r"AssertionError:", re.IGNORECASE),
    re.compile(r"AssertionError", re.IGNORECASE),  # catch misspellings in output
    re.compile(r"AssertionError", re.IGNORECASE),
    re.compile(r"assert .+ (==|!=|is|in) ", re.IGNORECASE),
    re.compile(r"Test Suites:.*failed", re.IGNORECASE),
    re.compile(r"Tests:\s+\d+ failed", re.IGNORECASE),
    re.compile(r"FAILURES", re.IGNORECASE),
    re.compile(r"pytest.*FAILED", re.IGNORECASE),
    re.compile(r"AssertionError", re.IGNORECASE),
]

_LINT_PATTERNS: list[re.Pattern] = [
    re.compile(r"eslint", re.IGNORECASE),
    re.compile(r"ruff\s+check", re.IGNORECASE),
    re.compile(r"flake8", re.IGNORECASE),
    re.compile(r"pylint", re.IGNORECASE),
    re.compile(r"\d+ problems? \(\d+ errors?, \d+ warnings?\)"),  # eslint summary
    re.compile(r"✖ \d+ problems?"),  # eslint summary (unicode variant)
    re.compile(r"Found \d+ errors?", re.IGNORECASE),  # ruff / generic
]

_TYPECHECK_PATTERNS: list[re.Pattern] = [
    re.compile(r"error TS\d+:"),  # tsc
    re.compile(r"error\[E\d+\]"),  # rustc
    re.compile(r"mypy.*error:", re.IGNORECASE),
    re.compile(r"pyright.*error:", re.IGNORECASE),
    re.compile(r"Type '.*' is not assignable"),
    re.compile(r"Property '.*' does not exist on type"),
    re.compile(r"has no attribute '.*'"),
]

_DEPENDENCY_PATTERNS: list[re.Pattern] = [
    re.compile(r"ModuleNotFoundError:", re.IGNORECASE),
    re.compile(r"ImportError:", re.IGNORECASE),
    re.compile(r"Cannot find module", re.IGNORECASE),
    re.compile(r"MODULE_NOT_FOUND", re.IGNORECASE),
    re.compile(r"No module named", re.IGNORECASE),
    re.compile(r"Could not resolve dependency", re.IGNORECASE),
    re.compile(r"ENOENT.*package\.json", re.IGNORECASE),
    re.compile(r"pip install", re.IGNORECASE),
    # Go module errors
    re.compile(r"no required module provides", re.IGNORECASE),
    re.compile(r"cannot find package .* in any of", re.IGNORECASE),
    re.compile(r"directory prefix \. does not contain main module", re.IGNORECASE),
    re.compile(r"go: no module file found", re.IGNORECASE),
    re.compile(r"missing go\.sum entry", re.IGNORECASE),
    re.compile(r"go: updates to go\.sum needed", re.IGNORECASE),
]

# File + line extractors (various compiler output formats)
_FILE_LINE_PATTERNS: list[re.Pattern] = [
    # TypeScript / ESLint: src/foo.ts(12,5): error ...
    re.compile(r"(\S+\.\w+)\((\d+),\d+\):\s*error"),
    # Python / generic: File "src/foo.py", line 12
    re.compile(r'File "([^"]+)", line (\d+)'),
    # GCC / Rust / Go: src/foo.go:12:5: error
    re.compile(r"(\S+\.\w+):(\d+):\d+:\s*error"),
    # ESLint: /path/to/file.ts:12:5
    re.compile(r"(\S+\.\w+):(\d+):\d+"),
    # Jest: at Object.<anonymous> (src/foo.test.ts:12:5)
    re.compile(r"\((\S+\.\w+):(\d+):\d+\)"),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_failure(output: str) -> FailureInfo:
    """Classify an error output string into a structured FailureInfo.

    Uses pattern matching — no LLM calls.  Order matters: more specific
    patterns (syntax, typecheck) are checked before broader ones (test,
    lint) so that e.g. a TypeScript error isn't misclassified as a generic
    test failure.
    """
    if not output:
        return FailureInfo(
            failure_type=UNKNOWN_FAILURE,
            summary="Empty error output",
            max_retries=1,
        )

    raw_snippet = output[:500]
    file_hint, line_hint = _extract_file_line(output)

    # --- Order: specific → general ---

    # 1. Syntax errors
    for pat in _SYNTAX_PATTERNS:
        if pat.search(output):
            return FailureInfo(
                failure_type=SYNTAX_ERROR,
                summary=_first_error_line(output, pat),
                raw_output=raw_snippet,
                file_hint=file_hint,
                line_hint=line_hint,
                max_retries=2,
            )

    # 2. Type-check errors (before lint — tsc errors look lint-ish)
    for pat in _TYPECHECK_PATTERNS:
        if pat.search(output):
            return FailureInfo(
                failure_type=TYPECHECK_FAILURE,
                summary=_first_error_line(output, pat),
                raw_output=raw_snippet,
                file_hint=file_hint,
                line_hint=line_hint,
                max_retries=2,
            )

    # 3. Missing dependency / import errors
    for pat in _DEPENDENCY_PATTERNS:
        if pat.search(output):
            return FailureInfo(
                failure_type=MISSING_DEPENDENCY,
                summary=_first_error_line(output, pat),
                raw_output=raw_snippet,
                file_hint=file_hint,
                line_hint=line_hint,
                max_retries=1,
            )

    # 4. Test failures
    for pat in _TEST_PATTERNS:
        if pat.search(output):
            return FailureInfo(
                failure_type=TEST_FAILURE,
                summary=_first_error_line(output, pat),
                raw_output=raw_snippet,
                file_hint=file_hint,
                line_hint=line_hint,
                max_retries=2,
            )

    # 5. Lint failures
    for pat in _LINT_PATTERNS:
        if pat.search(output):
            return FailureInfo(
                failure_type=LINT_FAILURE,
                summary=_first_error_line(output, pat),
                raw_output=raw_snippet,
                file_hint=file_hint,
                line_hint=line_hint,
                max_retries=2,
            )

    # 6. Fallback
    return FailureInfo(
        failure_type=UNKNOWN_FAILURE,
        summary=_first_meaningful_line(output),
        raw_output=raw_snippet,
        file_hint=file_hint,
        line_hint=line_hint,
        max_retries=1,
    )


def classify_review_rejection(review_text: str) -> FailureInfo:
    """Wrap a review rejection as a FailureInfo."""
    # Try to extract the most actionable line from the review
    summary = _first_meaningful_line(review_text)
    return FailureInfo(
        failure_type=REVIEW_REJECTION,
        summary=summary,
        raw_output=review_text[:500],
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Failure tracking for stop-condition detection
# ---------------------------------------------------------------------------

class FailureTracker:
    """Tracks repeated failures to decide when to stop retrying.

    A failure is considered "the same" if (failure_type, error_signature)
    matches a previously seen failure.  The error_signature is the first
    ~120 chars of the summary, normalized.
    """

    def __init__(self, max_repeats: int = 3):
        self.max_repeats = max_repeats
        # key = (failure_type, signature), value = count
        self._seen: dict[tuple[str, str], int] = {}
        self._history: list[FailureInfo] = []

    def record(self, info: FailureInfo) -> bool:
        """Record a failure.  Returns True if the agent should STOP retrying."""
        self._history.append(info)
        sig = _normalize_signature(info.summary)
        key = (info.failure_type, sig)
        self._seen[key] = self._seen.get(key, 0) + 1
        return self._seen[key] >= self.max_repeats

    @property
    def last(self) -> Optional[FailureInfo]:
        return self._history[-1] if self._history else None

    @property
    def dominant_failure_type(self) -> Optional[str]:
        """Return the most common failure type seen so far, or None."""
        if not self._history:
            return None
        counts: dict[str, int] = {}
        for info in self._history:
            counts[info.failure_type] = counts.get(info.failure_type, 0) + 1
        return max(counts, key=counts.get)

    def summary_for_storage(self) -> tuple[Optional[str], Optional[str]]:
        """Return (failure_type, failure_summary) for the dominant failure."""
        if not self._history:
            return None, None
        dominant = self.dominant_failure_type
        # Find the most recent instance of the dominant type
        for info in reversed(self._history):
            if info.failure_type == dominant:
                return info.failure_type, info.summary
        return dominant, self._history[-1].summary


# ---------------------------------------------------------------------------
# Retry guidance
# ---------------------------------------------------------------------------

# Maps failure_type → short instruction injected into the LLM prompt
RETRY_GUIDANCE: dict[str, str] = {
    SYNTAX_ERROR: (
        "Compilation or syntax error detected. Re-read the file near the error line "
        "and make a small targeted fix. Do NOT rewrite the whole file. "
        "For Go: '...' is not valid as a statement — replace with 'return nil' or "
        "an empty function body. 'undefined: pkg.X' for a third-party library → the "
        "API changed in the installed version. Find actual names with: "
        "`grep -rn '^type \\|^func ' $(go env GOPATH)/pkg/mod/<author>/<pkg>*/*.go 2>/dev/null | grep -v '_test.go' | head -40` "
        "(use glob * for version, e.g. github.com/asticode/go-astiav*). "
        "Many libraries also ship examples — check them: "
        "`ls $(go env GOPATH)/pkg/mod/<author>/<pkg>*/examples/ 2>/dev/null` "
        "then `cat` the relevant example to see the exact API usage pattern. "
        "Use `go doc <pkg>` or `go doc <pkg>.<Type>` to read method signatures. "
        "Use ONLY names that exist in the installed version. "
        "'undefined: X' (local) means X is out of scope or misspelled. "
        "'imported and not used' → remove the unused import line. "
        "'declared and not used: X' → remove variable X or use it. "
        "'cannot use X as type Y' or 'cannot convert X to type Y' → check the type "
        "definition in the source file and use the correct type or a proper conversion. "
        "Common Go type fixes: 'byte' → 'rune' requires rune(b) cast; "
        "'string' → '[]byte' requires []byte(s); 'int' → 'int64' requires int64(n). "
        "If indexing a string with s[i], the result is a byte, not a rune — "
        "use rune(s[i]) when a rune is needed. "
        "'cannot use *T as T' → declare variable as *T or dereference with (*result). "
        "'cannot use T as *T' → remove & or pass address. "
        "Read the function signature in the source to know if it returns T or *T. "
        "'invalid append: argument must be a slice; have result (variable of type *[]T)' → "
        "you are appending to a pointer-to-slice. Dereference first: "
        "`*result = append(*result, item)` or use a local slice variable. "
        "'cannot range over X (variable of type *[]T)' → same issue: dereference with `*X` before ranging."
    ),
    TEST_FAILURE: (
        "A test is failing. Read the failing test and the code it exercises. "
        "Make a focused fix to the production code (or the test setup if the test "
        "expectation is wrong). Re-run that specific test before moving on."
    ),
    LINT_FAILURE: (
        "Lint errors were reported. Fix ONLY the specific issues listed in the "
        "output. Do not refactor unrelated code."
    ),
    TYPECHECK_FAILURE: (
        "Type-check errors were reported. Fix the type annotations or usage that "
        "the checker flagged. Check existing type definitions before adding new ones."
    ),
    MISSING_DEPENDENCY: (
        "A missing module/dependency was reported. Check that the import path is "
        "correct. For Go: if 'does not contain main module' → go.mod must be in the "
        "CURRENT WORKING DIRECTORY (repo root). Move or create go.mod there with "
        "'go mod init <name>'. Then place all source files relative to that root. "
        "If 'no required module provides package X/Y/Z' → check if X is the module "
        "name in go.mod. If X is a long path like 'github.com/user/repo' but go.mod "
        "says 'module myrepo', your internal import is WRONG. Fix the import: replace "
        "the prefix with the actual module name from go.mod "
        "(e.g. import 'github.com/user/repo/internal/foo' → 'myrepo/internal/foo'). "
        "Only run 'go get <pkg>' for THIRD-PARTY packages not defined locally. "
        "If 'missing go.sum entry' → run 'go get ./...' then 'go mod tidy' from the "
        "directory containing go.mod. This updates both go.mod and go.sum. "
        "For Python/Node: only use packages already in requirements.txt/package.json."
    ),
    REVIEW_REJECTION: (
        "The reviewer found issues. Address each piece of feedback. Focus on what "
        "the reviewer specifically asked for — do not make unrelated changes."
    ),
    UNKNOWN_FAILURE: (
        "An unclassified error occurred. Read the error output carefully and "
        "attempt one diagnostic fix. If the same error repeats, stop and report."
    ),
}


def get_retry_guidance(failure_type: str, raw_output: str = "") -> str:
    """Return the retry instruction for a given failure type.

    If raw_output is provided, we extract specific recovery commands from it
    (e.g. 'to add it: go get pkg') and prepend them to the generic guidance.
    """
    base = RETRY_GUIDANCE.get(failure_type, RETRY_GUIDANCE[UNKNOWN_FAILURE])

    if not raw_output:
        return base

    # Extract 'go get <pkg>' commands suggested by the Go toolchain
    go_get_cmds = list(dict.fromkeys(  # deduplicate, preserve order
        m.group(0) for m in re.finditer(r"go get \S+", raw_output)
    ))
    # Extract 'go mod download <pkg>'
    go_mod_cmds = list(dict.fromkeys(
        m.group(0) for m in re.finditer(r"go mod download \S+", raw_output)
    ))

    specifics: list[str] = []
    if go_get_cmds:
        cmds = "  \n".join(go_get_cmds[:4])  # cap at 4 to avoid spam
        specifics.append(f"Run these commands first (suggested by compiler):\n  {cmds}")
    if go_mod_cmds:
        cmds = "  \n".join(go_mod_cmds[:4])
        specifics.append(f"Then run: {go_mod_cmds[0]}")

    if specifics:
        return "\n".join(specifics) + "\n" + base
    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_file_line(output: str) -> tuple[Optional[str], Optional[int]]:
    """Try to extract a filename and line number from error output."""
    for pat in _FILE_LINE_PATTERNS:
        m = pat.search(output)
        if m:
            try:
                return m.group(1), int(m.group(2))
            except (IndexError, ValueError):
                return m.group(1), None
    return None, None


def _first_error_line(output: str, pattern: re.Pattern) -> str:
    """Return the first line matching the pattern, truncated."""
    for line in output.splitlines():
        if pattern.search(line):
            return line.strip()[:150]
    return output.splitlines()[0][:150] if output else ""


def _first_meaningful_line(text: str) -> str:
    """Return the first non-empty, non-whitespace line."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 5:
            return stripped[:150]
    return text[:150] if text else ""


def _normalize_signature(summary: str) -> str:
    """Normalize an error summary for deduplication.

    Strips numbers, paths, and whitespace so that the "same" error
    with different line numbers still matches.
    """
    s = summary.lower()
    s = re.sub(r"\d+", "#", s)  # Replace numbers
    s = re.sub(r"['\"].*?['\"]", "'?'", s)  # Replace quoted strings
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]