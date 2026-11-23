"""Unit tests for `.github/scripts/check_ratchets.py` and `.github/scripts/trial_merge.py`."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS_DIR / name
    mod_name = f"_ticktack_ci_{name.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


check_ratchets = _load_script("check_ratchets.py")
trial_merge = _load_script("trial_merge.py")


def test_code_line_numbers_excludes_docstrings_comments_and_blanks() -> None:
    sample = (
        '"""Module docstring.\n'
        'Second line.\n'
        '"""\n'
        "\n"
        "# Full-line comment\n"
        "x = 1  # inline comment still counts as code\n"
        "\n"
        "def f(a):\n"
        '    """Function docstring."""\n'
        "    # indented comment\n"
        "    return a + 1\n"
    )
    assert check_ratchets.code_line_numbers(sample) == {6, 8, 11}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Line-budget-reason: Added Lyapunov covariance helper for Gate 2.", "Added Lyapunov covariance helper for Gate 2."),
        ("- line-budget-reason:   Split solver helper cleanly  ", "Split solver helper cleanly"),
        ("<!-- Line-budget-reason: hidden -->\nLine-budget-reason:   ", None),
        ("Line-budget-reason: N/A", None),
        ("Line-budget-reason: <required if >0>", None),
        ("Line-budget-reason: [fill me in]", None),
        ("No reason line at all.", None),
    ],
)
def test_extract_line_budget_reason(body: str, expected: str | None) -> None:
    assert check_ratchets.extract_line_budget_reason(body) == expected


def test_parse_ruff_complexity_json_filters_selected_rules() -> None:
    payload = json.dumps(
        [
            {"code": "C901", "filename": "src/ticktack/a.py"},
            {"code": "PLR0913", "filename": "src/ticktack/a.py"},
            {"code": "PLR0913", "filename": "src/ticktack/b.py"},
            {"code": "E501", "filename": "src/ticktack/b.py"},
        ]
    )
    total, by_code = check_ratchets.parse_ruff_complexity_json(
        payload, ["C901", "PLR0912", "PLR0913", "PLR0915"]
    )
    assert total == 3
    assert by_code == {"C901": 1, "PLR0912": 0, "PLR0913": 2, "PLR0915": 0}


def test_parse_diff_hunk_lines_handles_add_modify_delete() -> None:
    diff_text = (
        "diff --git a/src/ticktack/foo.py b/src/ticktack/foo.py\n"
        "--- a/src/ticktack/foo.py\n"
        "+++ b/src/ticktack/foo.py\n"
        "@@ -10,2 +10,3 @@ def bar():\n"
        "-old1\n"
        "-old2\n"
        "+new1\n"
        "+new2\n"
        "+new3\n"
        "@@ -30 +31,0 @@\n"
        "-deleted_single\n"
    )
    parsed = check_ratchets.parse_diff_hunk_lines(diff_text)
    assert parsed == {"src/ticktack/foo.py": ({10, 11, 30}, {10, 11, 12})}


def test_check_ratchets_line_budget_enforcement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "src" / "ticktack"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    cfg = tmp_path / "ci_ratchets.json"
    cfg.write_text(
        json.dumps(
            {
                "complexity_rules": ["C901", "PLR0912", "PLR0913", "PLR0915"],
                "max_complexity_violations": 2,
                "max_src_code_lines": 2,
                "max_pr_changed_src_lines": 400,
                "min_coverage_percent": 50,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        check_ratchets,
        "run_complexity_check",
        lambda src_dir, rules, ruff_bin="ruff": (2, {"C901": 2}),
    )

    # Fails without Line-budget-reason
    opts_bad = check_ratchets.RatchetOptions(pr_body="Line-budget-reason: N/A")
    assert check_ratchets.check_ratchets(tmp_path, cfg, opts_bad) == 1
    # Passes with a real justification
    opts_ok = check_ratchets.RatchetOptions(
        pr_body="Line-budget-reason: Adds required numerical guard."
    )
    assert check_ratchets.check_ratchets(tmp_path, cfg, opts_ok) == 0
    # Fails when complexity exceeds baseline even with Line-budget-reason
    monkeypatch.setattr(
        check_ratchets,
        "run_complexity_check",
        lambda src_dir, rules, ruff_bin="ruff": (3, {"C901": 3}),
    )
    assert check_ratchets.check_ratchets(tmp_path, cfg, opts_ok) == 1


def test_select_overlapping_prs_excludes_self_and_stacked_bases() -> None:
    open_prs = [
        {
            "number": 56,
            "title": "PR 56",
            "baseRefName": "main",
            "files": [{"path": "src/ticktack/likelihood.py"}, {"path": "src/ticktack/fields.py"}],
        },
        {
            "number": 57,
            "title": "PR 57",
            "baseRefName": "main",
            "files": [{"path": "src/ticktack/likelihood.py"}, {"path": "src/ticktack/fisher.py"}],
        },
        {
            "number": 58,
            "title": "Stacked PR 58",
            "baseRefName": "v2/subspace-gp-core",
            "files": [{"path": "src/ticktack/likelihood.py"}],
        },
        {
            "number": 60,
            "title": "Disjoint PR 60",
            "baseRefName": "main",
            "files": [{"path": "tests/test_preset_steady_state.py"}],
        },
    ]
    overlapping, skipped = trial_merge.select_overlapping_prs(
        open_prs,
        current_pr=56,
        base_ref="main",
        current_files={"src/ticktack/likelihood.py", "src/ticktack/fields.py"},
    )
    assert [(int(pr["number"]), files) for pr, files in overlapping] == [
        (57, ("src/ticktack/likelihood.py",))
    ]
    assert [int(pr["number"]) for pr in skipped] == [58]


def test_count_pr_changed_src_lines_ignores_docstrings_and_comments(tmp_path: Path) -> None:
    """Verify git diff AST code-line counter excludes docstrings, comments, and blank lines."""
    import subprocess

    def _git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True)

    _git("init", "-b", "main")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@example.com")
    pkg = tmp_path / "src" / "ticktack"
    pkg.mkdir(parents=True)
    mod = pkg / "sample.py"
    mod.write_text('"""Doc."""\nx = 1\ny = 2\n', encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")

    _git("checkout", "-b", "feature")
    mod.write_text(
        '"""Expanded docstring.\n\nMore prose.\n"""\n# comment line\nx = 1\ny = 3\nz = 4\n',
        encoding="utf-8",
    )
    _git("add", ".")
    _git("commit", "-m", "feature")

    changed, touched = check_ratchets.count_pr_changed_src_lines(tmp_path, "main")
    # Removed code line `y = 2` (1) + added code lines `y = 3`, `z = 4` (2) = 3
    assert changed == 3
    assert touched == ["src/ticktack/sample.py"]


def test_repo_ratchets_config_matches_baseline() -> None:
    """Verify `.github/ci_ratchets.json` schema and baseline line count on the repo."""
    cfg_path = REPO_ROOT / ".github" / "ci_ratchets.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in (
        "baseline_commit",
        "complexity_rules",
        "max_complexity_violations",
        "max_src_code_lines",
        "max_pr_changed_src_lines",
        "claude_review_line_threshold",
        "claude_review_modules",
        "min_coverage_percent",
    ):
        assert key in cfg, f"Missing required key {key!r} in .github/ci_ratchets.json"

    total_lines, _ = check_ratchets.count_src_code_lines(REPO_ROOT / "src" / "ticktack")
    assert total_lines <= int(cfg["max_src_code_lines"])


def test_ci_scripts_obey_function_shape_limits() -> None:
    """Every function in `.github/scripts/*.py` must be <=80 code lines and <=8 parameters."""
    for script in sorted(SCRIPTS_DIR.glob("*.py")):
        source = script.read_text(encoding="utf-8")
        code_lines = check_ratchets.code_line_numbers(source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assert node.end_lineno is not None
            n_code = sum(1 for ln in range(node.lineno, node.end_lineno + 1) if ln in code_lines)
            args = node.args
            params = [
                a.arg
                for a in (args.posonlyargs + args.args + args.kwonlyargs)
                if a.arg not in ("self", "cls")
            ]
            if args.vararg:
                params.append(args.vararg.arg)
            if args.kwarg:
                params.append(args.kwarg.arg)
            assert n_code <= 80, f"{script.name}:{node.name} has {n_code} code lines (>80)"
            assert len(params) <= 8, f"{script.name}:{node.name} has {len(params)} params (>8)"
