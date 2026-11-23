"""Deterministic CI ratchets for ticktack (Section 3 review gates).

Checks:
1. Ruff complexity ratchet (C901, PLR0912, PLR0913, PLR0915) against baseline count.
2. AST source code line budget across ``src/ticktack/**/*.py`` (matching ``shape.py``).
3. Carbon shape-exemption comment validation (``# shape-exempt(carbon): <reason>``).
4. Optional PR changed-source-line scope warning, Gate 5 Claude-review notice,
   and new/edited function shape limits (<=80 AST code lines, <=8 parameters).
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

DEFAULT_MAX_FUNCTION_CODE_LINES = 80
DEFAULT_MAX_FUNCTION_PARAMS = 8
MIN_SHAPE_EXEMPT_REASON_LEN = 15
SHAPE_EXEMPT_PATTERN = re.compile(
    r"shape-exempt\(\s*carbon\s*\)\s*(?::\s*(.*))?$",
    re.IGNORECASE,
)
TRAILING_LINT_DIRECTIVE = re.compile(r"\s*#\s*(?:noqa|type:\s*ignore).*$", re.IGNORECASE)
VACUOUS_SHAPE_TOKENS = frozenset(
    {
        "a", "an", "and", "approved", "are", "arg", "args", "argument", "arguments",
        "as", "at", "avoid", "avoiding", "avoids", "awkward", "be", "because",
        "better", "break", "breaking", "breaks", "by", "carbon", "clarity", "clean",
        "cleaner", "cleanest", "clear", "clearer", "code", "cohesion", "cohesive",
        "complex", "complexity", "complicated", "contort", "contortion", "convenience",
        "convenient", "do", "does", "easier", "easy", "exempt", "exemption", "for",
        "from", "function", "good", "has", "have", "helper", "helpers", "here",
        "hurt", "hurting", "hurts", "i", "in", "inline", "inlined", "is", "it",
        "its", "just", "keep", "keeping", "kept", "legacy", "less", "like", "limit",
        "limits", "line", "lines", "logic", "long", "maintainability", "maintainable",
        "make", "makes", "making", "many", "method", "more", "most", "much", "my",
        "natural", "naturally", "necessary", "need", "needed", "needs", "nice", "no",
        "not", "of", "on", "one", "or", "other", "our", "out", "over", "param",
        "parameter", "parameters", "params", "piece", "place", "pragmatic", "read",
        "readability", "readable", "really", "reason", "refactor", "refactoring",
        "same", "see", "shape", "should", "simple", "simpler", "size", "so", "some",
        "split", "splitting", "stay", "stays", "stuff", "style", "that", "the",
        "their", "them", "there", "these", "thing", "things", "this", "tidy", "to",
        "together", "too", "understand", "understandable", "very", "want", "way",
        "we", "well", "with", "worse", "would",
    }
)


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


def _git_merge_base(repo_root: Path, base_ref: str) -> str:
    proc = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git merge-base against {base_ref} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _git_head_commit_body(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%B"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def count_function_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count function parameters excluding ``self`` and ``cls`` (including ``*args``/``**kwargs``)."""
    args = node.args
    names = [
        a.arg
        for a in (args.posonlyargs + args.args + args.kwonlyargs)
        if a.arg not in ("self", "cls")
    ]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return len(names)


def validate_shape_exemption_reason(reason: str | None) -> str | None:
    """Validate a ``# shape-exempt(carbon): <reason>`` string; return error message or None."""
    if reason is None:
        return "missing ':' and technical reason"
    cleaned = TRAILING_LINT_DIRECTIVE.sub("", reason).strip()
    core = cleaned.rstrip(" .!;,-")
    if (
        not core
        or core.lower() in PLACEHOLDER_REASONS
        or (core.startswith("<") and core.endswith(">"))
        or (core.startswith("[") and core.endswith("]"))
    ):
        return "placeholder or empty reason"
    if len(core) < MIN_SHAPE_EXEMPT_REASON_LEN:
        return f"reason too short ({len(core)} < {MIN_SHAPE_EXEMPT_REASON_LEN} chars): {core!r}"
    tokens = re.findall(r"[a-z0-9_]+", core.lower())
    specific = {
        tok for tok in tokens if tok not in VACUOUS_SHAPE_TOKENS and (len(tok) >= 2 or tok.isdigit())
    }
    if len(specific) < 2:
        return f"vacuous or non-specific reason (needs concrete technical detail): {core!r}"
    return None


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(getattr(stmt, "value", None), ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _def_header_end_lines(
    tokens: list[tokenize.TokenInfo],
    def_linenos: set[int],
) -> dict[int, int]:
    """Map each ``def`` start line to the line containing its header-closing ``:``."""
    header_ends: dict[int, int] = {}
    for idx, tok in enumerate(tokens):
        if tok.type != tokenize.NAME or tok.string != "def" or tok.start[0] not in def_linenos:
            continue
        depth = 0
        seen_paren = False
        for nxt in tokens[idx + 1 :]:
            if nxt.type == tokenize.OP:
                if nxt.string in "([{":
                    depth += 1
                    if nxt.string == "(":
                        seen_paren = True
                elif nxt.string in ")]}":
                    depth = max(0, depth - 1)
                elif nxt.string == ":" and depth == 0 and seen_paren:
                    header_ends[tok.start[0]] = nxt.start[0]
                    break
    return header_ends


def _function_exemption_zones(
    lines: list[str],
    tree: ast.AST,
    tokens: list[tokenize.TokenInfo],
) -> list[tuple[int, set[int], set[int], set[int]]]:
    """Return ``(def_lineno, header_lines, above_lines, inside_lines)`` for each function."""
    funcs = sorted(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda n: n.lineno,
    )
    header_ends = _def_header_end_lines(tokens, {n.lineno for n in funcs})
    zones: list[tuple[int, set[int], set[int], set[int]]] = []
    for node in funcs:
        deco_start = min((d.lineno for d in node.decorator_list), default=node.lineno)
        header_end = header_ends.get(node.lineno, node.lineno)
        header_lines = set(range(deco_start, header_end + 1))

        above_lines: set[int] = set()
        ln = deco_start - 1
        while ln >= 1 and lines[ln - 1].strip().startswith("#"):
            above_lines.add(ln)
            ln -= 1

        inside_lines: set[int] = set()
        if node.body:
            first = node.body[0]
            if first.lineno > header_end:
                inside_lines.update(range(header_end + 1, first.lineno))
            if _is_docstring_stmt(first) and first.end_lineno is not None:
                if len(node.body) > 1:
                    inside_lines.update(range(first.end_lineno + 1, node.body[1].lineno))
                elif node.end_lineno is not None:
                    inside_lines.update(range(first.end_lineno + 1, node.end_lineno + 1))
        zones.append((node.lineno, header_lines, above_lines, inside_lines))
    return zones


def _match_exemption_line(
    tok_line: int,
    zones: list[tuple[int, set[int], set[int], set[int]]],
) -> int | None:
    """Attach a comment line to its target ``def`` (precedence: header > above > inside)."""
    for tier in (1, 2, 3):
        matches = [z[0] for z in zones if tok_line in z[tier]]
        if matches:
            return max(matches)
    return None


def extract_shape_exemptions(
    source: str,
    filename: str = "<string>",
) -> tuple[dict[int, str], list[str]]:
    """Parse and validate ``# shape-exempt(carbon): <reason>`` comments in ``source``."""
    tree = ast.parse(source)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    zones = _function_exemption_zones(source.split("\n"), tree, tokens)

    exempt: dict[int, str] = {}
    errors: list[str] = []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        match = SHAPE_EXEMPT_PATTERN.search(tok.string)
        if not match:
            continue
        tok_line = tok.start[0]
        raw_reason = match.group(1)
        reason_err = validate_shape_exemption_reason(raw_reason)
        if reason_err is not None:
            errors.append(f"{filename}:{tok_line}: invalid # shape-exempt(carbon): {reason_err}")
        target_def = _match_exemption_line(tok_line, zones)
        if target_def is None:
            errors.append(
                f"{filename}:{tok_line}: # shape-exempt(carbon) must be placed on, "
                "immediately above, or immediately inside a def"
            )
        elif reason_err is None and raw_reason is not None:
            exempt[target_def] = TRAILING_LINT_DIRECTIVE.sub("", raw_reason).strip()
    return exempt, errors


def check_shape_exemptions(src_dir: Path) -> list[str]:
    """Validate all ``# shape-exempt(carbon)`` comments across ``src_dir``."""
    errors: list[str] = []
    for path in sorted(src_dir.rglob("*.py")):
        rel = path.relative_to(src_dir.parent).as_posix()
        _, file_errors = extract_shape_exemptions(path.read_text(encoding="utf-8"), filename=rel)
        errors.extend(file_errors)
    return errors


def _diff_src_hunks(
    repo_root: Path,
    base_ref: str,
) -> tuple[str, dict[str, tuple[set[int], set[int]]]]:
    merge_base = _git_merge_base(repo_root, base_ref)
    proc = subprocess.run(
        ["git", "diff", "-U0", "--no-renames", f"{merge_base}..HEAD", "--", "src/ticktack"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git diff against {base_ref} failed: {proc.stderr.strip()}")
    return merge_base, parse_diff_hunk_lines(proc.stdout)


def count_pr_changed_src_lines(repo_root: Path, base_ref: str) -> tuple[int, list[str]]:
    """Count changed AST code lines (added + removed) under ``src/ticktack/`` against ``base_ref``."""
    merge_base, per_file = _diff_src_hunks(repo_root, base_ref)
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
            base_src = _git_show_text(repo_root, merge_base, rel_path)
            if base_src is not None:
                total_changed += len(removed & code_line_numbers(base_src))
    return total_changed, touched_py


def _qualified_functions(
    tree: ast.AST,
) -> list[tuple[tuple[str, int], ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Return ``((qualname, occurrence_idx), node)`` for every function in ``tree``."""
    found: list[tuple[tuple[str, int], ast.FunctionDef | ast.AsyncFunctionDef]] = []
    seen: dict[str, int] = {}

    def _visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                cls_prefix = f"{prefix}.{child.name}" if prefix else child.name
                _visit(child, cls_prefix)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = f"{prefix}.{child.name}" if prefix else child.name
                occ = seen.get(qname, 0)
                seen[qname] = occ + 1
                found.append(((qname, occ), child))
                _visit(child, f"{qname}.<locals>")
            else:
                _visit(child, prefix)

    _visit(tree, "")
    return found


def _base_function_shapes(base_src: str | None) -> dict[tuple[str, int], tuple[int, int]]:
    """Map qualified function keys in ``base_src`` to ``(code_lines, param_count)``."""
    if base_src is None:
        return {}
    base_code = code_line_numbers(base_src)
    shapes: dict[tuple[str, int], tuple[int, int]] = {}
    for key, node in _qualified_functions(ast.parse(base_src)):
        end_ln = node.end_lineno or node.lineno
        n_code = sum(1 for ln in range(node.lineno, end_ln + 1) if ln in base_code)
        shapes[key] = (n_code, count_function_params(node))
    return shapes


def check_changed_function_shapes(
    repo_root: Path,
    base_ref: str,
    max_lines: int = DEFAULT_MAX_FUNCTION_CODE_LINES,
    max_params: int = DEFAULT_MAX_FUNCTION_PARAMS,
) -> list[str]:
    """Check new or edited functions under ``src/ticktack/`` against shape limits.

    Pre-existing legacy functions exceeding ``(max_lines, max_params)`` on the merge-base
    get a monotonic ceiling ``(max(max_lines, base_lines), max(max_params, base_params))``
    so bugfixes and incremental shrink refactors pass while any growth fails unless
    Carbon-exempted.
    """
    merge_base, per_file = _diff_src_hunks(repo_root, base_ref)
    failures: list[str] = []
    for rel_path, (_, added) in sorted(per_file.items()):
        if not rel_path.endswith(".py") or not added:
            continue
        head_file = repo_root / rel_path
        if not head_file.exists():
            continue
        source = head_file.read_text(encoding="utf-8")
        code_lines = code_line_numbers(source)
        added_code = added & code_lines
        if not added_code:
            continue
        exempt_by_line, _ = extract_shape_exemptions(source, filename=rel_path)
        base_shapes = _base_function_shapes(_git_show_text(repo_root, merge_base, rel_path))
        for key, node in _qualified_functions(ast.parse(source)):
            deco_start = min((d.lineno for d in node.decorator_list), default=node.lineno)
            end_lineno = node.end_lineno or node.lineno
            if not any(deco_start <= ln <= end_lineno for ln in added_code):
                continue
            n_code = sum(1 for ln in range(node.lineno, end_lineno + 1) if ln in code_lines)
            n_params = count_function_params(node)
            base_lines, base_params = base_shapes.get(key, (0, 0))
            eff_max_lines = max(max_lines, base_lines)
            eff_max_params = max(max_params, base_params)
            if n_code <= eff_max_lines and n_params <= eff_max_params:
                continue
            qname = key[0]
            if node.lineno in exempt_by_line:
                print(
                    f"Shape exemption (carbon) accepted for {rel_path}:{node.lineno} "
                    f"{qname} ({n_code} lines, {n_params} params): {exempt_by_line[node.lineno]}"
                )
                continue
            failures.append(
                f"Function shape limit exceeded: {rel_path}:{node.lineno} {qname} has "
                f"{n_code} code lines (limit {eff_max_lines}) and {n_params} params "
                f"(limit {eff_max_params}). Refactor cleanly, or obtain Carbon reviewer "
                "approval and add '# shape-exempt(carbon): <specific >=15-char technical reason>'."
            )
    return failures


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


def _load_pr_body(repo_root: Path, pr_body_file: Path | None) -> str:
    if pr_body_file is not None and pr_body_file.exists():
        return pr_body_file.read_text(encoding="utf-8")
    env_body = os.environ.get("PR_BODY", "")
    if env_body.strip():
        return env_body
    return _git_head_commit_body(repo_root)


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

    failures.extend(check_shape_exemptions(pkg_dir))

    if opts.base_ref:
        failures.extend(_check_pr_scope(repo_root, opts.base_ref, cfg, opts.fail_on_pr_size))
        max_fn_lines = int(cfg.get("max_function_code_lines", DEFAULT_MAX_FUNCTION_CODE_LINES))
        max_fn_params = int(cfg.get("max_function_params", DEFAULT_MAX_FUNCTION_PARAMS))
        failures.extend(
            check_changed_function_shapes(
                repo_root, opts.base_ref, max_lines=max_fn_lines, max_params=max_fn_params
            )
        )

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
        pr_body=_load_pr_body(args.repo_root, args.pr_body_file),
        fail_on_pr_size=args.fail_on_pr_size,
    )
    return check_ratchets(repo_root=args.repo_root, config_path=config_path, options=opts)


if __name__ == "__main__":
    raise SystemExit(main())
