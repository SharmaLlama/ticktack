"""Deterministic CI ratchets for ticktack (Section 3 review gates).

Checks:
1. Ruff complexity ratchet (C901, PLR0912, PLR0913, PLR0915) against baseline count.
2. AST source code line budget across ``src/ticktack/**/*.py`` (matching ``shape.py``).
3. Optional PR changed-source-line scope warning and Gate 5 Claude-review notice.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RatchetOptions:
    """Runtime options for ``check_ratchets``."""

    ruff_bin: str = "ruff"
    base_ref: str | None = None
    pr_body: str = ""
    fail_on_pr_size: bool = False

PLACEHOLDER_REASONS = frozenset({"", "none", "n/a", "na", "tbd", "todo", "...", "-"})
REASON_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?Line-budget-reason:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
HUNK_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def code_line_numbers(source: str) -> set[int]:
    """Return 1-based line numbers containing executable code (no blanks/comments/docstrings)."""
    lines = source.split("\n")
    tree = ast.parse(source)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
                and body[0].end_lineno is not None
            ):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))

    comment_lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT and lines[tok.start[0] - 1].strip().startswith("#"):
            comment_lines.add(tok.start[0])

    return {
        idx
        for idx, line in enumerate(lines, 1)
        if line.strip() and idx not in doc_lines and idx not in comment_lines
    }


def count_src_code_lines(src_dir: Path) -> tuple[int, dict[str, int]]:
    """Count AST code lines across all Python files under ``src_dir``."""
    per_file: dict[str, int] = {}
    for path in sorted(src_dir.rglob("*.py")):
        rel = path.relative_to(src_dir).as_posix()
        per_file[rel] = len(code_line_numbers(path.read_text(encoding="utf-8")))
    return sum(per_file.values()), per_file


def extract_line_budget_reason(pr_body: str) -> str | None:
    """Return a non-placeholder ``Line-budget-reason:`` explanation from ``pr_body``, or None."""
    cleaned = HTML_COMMENT_PATTERN.sub("", pr_body)
    for match in REASON_PATTERN.finditer(cleaned):
        candidate = match.group(1).strip()
        if candidate.lower() in PLACEHOLDER_REASONS:
            continue
        if (candidate.startswith("<") and candidate.endswith(">")) or (
            candidate.startswith("[") and candidate.endswith("]")
        ):
            continue
        return candidate
    return None


def parse_ruff_complexity_json(stdout: str, rules: list[str]) -> tuple[int, dict[str, int]]:
    """Parse JSON output from ``ruff check --output-format json``."""
    items = json.loads(stdout)
    allowed = set(rules)
    by_code: dict[str, int] = {code: 0 for code in rules}
    total = 0
    for item in items:
        code = item.get("code")
        if code in allowed:
            by_code[code] = by_code.get(code, 0) + 1
            total += 1
    return total, by_code


def run_complexity_check(
    src_dir: Path,
    rules: list[str],
    ruff_bin: str = "ruff",
) -> tuple[int, dict[str, int]]:
    """Run ruff complexity rules against ``src_dir`` and return ``(total, per_code)``."""
    cmd = [
        ruff_bin,
        "check",
        str(src_dir),
        "--select",
        ",".join(rules),
        "--output-format",
        "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"ruff failed with exit code {proc.returncode}: {proc.stderr.strip()}"
        )
    return parse_ruff_complexity_json(proc.stdout, rules)


def parse_diff_hunk_lines(diff_text: str) -> dict[str, tuple[set[int], set[int]]]:
    """Map repo-relative file paths to ``(removed_base_lines, added_head_lines)``."""
    changed: dict[str, tuple[set[int], set[int]]] = {}
    current_path: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            current_path = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else None
            if current_path is not None:
                changed.setdefault(current_path, (set(), set()))
            continue
        if current_path is None:
            continue
        match = HUNK_PATTERN.match(line)
        if not match:
            continue
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) is not None else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) is not None else 1
        removed, added = changed[current_path]
        if old_count > 0:
            removed.update(range(old_start, old_start + old_count))
        if new_count > 0:
            added.update(range(new_start, new_start + new_count))
    return changed


def _git_show_text(repo_root: Path, ref: str, rel_path: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def count_pr_changed_src_lines(repo_root: Path, base_ref: str) -> tuple[int, list[str]]:
    """Count changed AST code lines (added + removed) under ``src/ticktack/`` against ``base_ref``."""
    proc = subprocess.run(
        ["git", "diff", "-U0", "--no-renames", f"{base_ref}...HEAD", "--", "src/ticktack"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git diff against {base_ref} failed: {proc.stderr.strip()}")

    per_file = parse_diff_hunk_lines(proc.stdout)
    total_changed = 0
    touched_py: list[str] = []
    for rel_path, (removed, added) in sorted(per_file.items()):
        if not rel_path.endswith(".py"):
            continue
        touched_py.append(rel_path)
        head_file = repo_root / rel_path
        if head_file.exists() and added:
            total_changed += len(added & code_line_numbers(head_file.read_text(encoding="utf-8")))
        if removed:
            base_src = _git_show_text(repo_root, base_ref, rel_path)
            if base_src is not None:
                total_changed += len(removed & code_line_numbers(base_src))
    return total_changed, touched_py


def _check_pr_scope(
    repo_root: Path,
    base_ref: str,
    cfg: dict[str, object],
    fail_on_pr_size: bool,
) -> list[str]:
    changed, touched = count_pr_changed_src_lines(repo_root, base_ref)
    max_changed = int(cfg["max_pr_changed_src_lines"])  # type: ignore[arg-type]
    print(f"PR changed src/ticktack code lines vs {base_ref}: {changed} (limit {max_changed})")
    failures: list[str] = []
    if changed > max_changed:
        msg = (
            f"PR changes {changed} src/ticktack code lines, exceeding the "
            f"{max_changed}-line single-PR scope guideline."
        )
        print(f"::warning::{msg}")
        if fail_on_pr_size:
            failures.append(msg)

    gate5_modules = set(cfg.get("claude_review_modules", []))  # type: ignore[arg-type]
    gate5_hits = sorted(gate5_modules.intersection(touched))
    gate5_thresh = int(cfg.get("claude_review_line_threshold", 200))  # type: ignore[arg-type]
    if gate5_hits or changed > gate5_thresh:
        print(
            "::notice::Section 3 Gate 5 applies (needs-claude-review): "
            f"core modules={gate5_hits or 'none'}, changed_src_lines={changed}."
        )
    return failures


def _load_pr_body(pr_body_file: Path | None) -> str:
    if pr_body_file is not None and pr_body_file.exists():
        return pr_body_file.read_text(encoding="utf-8")
    return os.environ.get("PR_BODY", "")


def check_ratchets(
    repo_root: Path,
    config_path: Path,
    options: RatchetOptions | None = None,
) -> int:
    """Execute all configured ratchets and return 0 on pass or 1 on failure."""
    opts = options or RatchetOptions()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    src_root = repo_root / "src"
    pkg_dir = src_root / "ticktack"
    failures: list[str] = []

    rules = list(cfg["complexity_rules"])
    max_complexity = int(cfg["max_complexity_violations"])
    total_cx, by_code = run_complexity_check(src_root, rules, ruff_bin=opts.ruff_bin)
    print(f"Complexity violations: {total_cx} / {max_complexity} ({by_code})")
    if total_cx > max_complexity:
        failures.append(
            f"Complexity ratchet failed: {total_cx} violations exceed baseline {max_complexity}."
        )

    max_lines = int(cfg["max_src_code_lines"])
    total_lines, _ = count_src_code_lines(pkg_dir)
    delta_lines = total_lines - max_lines
    print(f"Source code lines (src/ticktack/**/*.py): {total_lines} / {max_lines} (delta {delta_lines:+d})")
    if total_lines > max_lines:
        reason = extract_line_budget_reason(opts.pr_body)
        if reason:
            print(f"Line budget increase justified ({delta_lines:+d} lines): {reason}")
        else:
            failures.append(
                f"Line budget exceeded: {total_lines} > {max_lines} ({delta_lines:+d} lines). "
                "Reduce source lines or add a non-empty 'Line-budget-reason: ...' line to the PR body."
            )

    if opts.base_ref:
        failures.extend(_check_pr_scope(repo_root, opts.base_ref, cfg, opts.fail_on_pr_size))

    for err in failures:
        print(f"ERROR: {err}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for CI ratchets."""
    parser = argparse.ArgumentParser(description="Run ticktack CI complexity and line-budget ratchets.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path(".github/ci_ratchets.json"))
    parser.add_argument("--ruff-bin", default="ruff")
    parser.add_argument("--base-ref", default=None)
    parser.add_argument("--pr-body-file", type=Path, default=None)
    parser.add_argument("--fail-on-pr-size", action="store_true")
    parser.add_argument("--print-coverage-floor", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else args.repo_root / args.config
    if args.print_coverage_floor:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        print(cfg["min_coverage_percent"])
        return 0

    opts = RatchetOptions(
        ruff_bin=args.ruff_bin,
        base_ref=args.base_ref,
        pr_body=_load_pr_body(args.pr_body_file),
        fail_on_pr_size=args.fail_on_pr_size,
    )
    return check_ratchets(repo_root=args.repo_root, config_path=config_path, options=opts)


if __name__ == "__main__":
    raise SystemExit(main())
