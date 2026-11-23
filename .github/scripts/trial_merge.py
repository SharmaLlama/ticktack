"""Trial-merge overlapping open PRs against the current PR to catch silent semantic clashes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialOutcome:
    """Result of trial-merging one overlapping open PR into the current PR."""

    pr_number: int
    title: str
    overlap: tuple[str, ...]
    merge_status: str
    test_status: str


def list_open_prs(repo: str) -> list[dict[str, object]]:
    """Query open pull requests via the GitHub CLI."""
    cmd = [
        "gh",
        "pr",
        "list",
        "-R",
        repo,
        "--state",
        "open",
        "--limit",
        "100",
        "--json",
        "number,title,baseRefName,headRefName,headRefOid,files",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh pr list failed ({proc.returncode}): {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def pr_touched_paths(pr_record: dict[str, object]) -> set[str]:
    """Extract changed file paths from a ``gh pr list --json files`` record."""
    raw_files = pr_record.get("files") or []
    paths: set[str] = set()
    for entry in raw_files:  # type: ignore[union-attr]
        if isinstance(entry, dict) and entry.get("path"):
            paths.add(str(entry["path"]))
    return paths


def select_overlapping_prs(
    open_prs: list[dict[str, object]],
    current_pr: int,
    base_ref: str,
    current_files: set[str],
) -> tuple[list[tuple[dict[str, object], tuple[str, ...]]], list[dict[str, object]]]:
    """Return ``(overlapping_same_base_prs, skipped_other_base_prs)``."""
    overlapping: list[tuple[dict[str, object], tuple[str, ...]]] = []
    skipped_other_base: list[dict[str, object]] = []
    for pr in open_prs:
        num = int(pr["number"])  # type: ignore[arg-type]
        if num == current_pr:
            continue
        if str(pr.get("baseRefName", "")) != base_ref:
            skipped_other_base.append(pr)
            continue
        shared = sorted(current_files.intersection(pr_touched_paths(pr)))
        if shared:
            overlapping.append((pr, tuple(shared)))
    overlapping.sort(key=lambda pair: int(pair[0]["number"]))  # type: ignore[arg-type]
    return overlapping, skipped_other_base


def current_pr_changed_files(repo_root: Path, base_ref: str) -> set[str]:
    """Return paths modified by HEAD relative to ``origin/<base_ref>``."""
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git diff --name-only failed: {proc.stderr.strip()}")
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _trial_merge_one(
    repo_root: Path,
    head_sha: str,
    pr: dict[str, object],
    overlap: tuple[str, ...],
) -> TrialOutcome:
    other_num = int(pr["number"])  # type: ignore[arg-type]
    title = str(pr.get("title", ""))
    print(f"\n=== Trial merge with PR #{other_num}: {title} ===")
    print(f"Overlapping files ({len(overlap)}): {', '.join(overlap)}")

    checkout = _run_git(repo_root, ["checkout", "--detach", "--force", head_sha])
    if checkout.returncode != 0:
        raise RuntimeError(f"git checkout --detach {head_sha} failed: {checkout.stderr}")
    _run_git(repo_root, ["clean", "-fd"])

    fetch = _run_git(repo_root, ["fetch", "origin", f"pull/{other_num}/head"])
    if fetch.returncode != 0:
        raise RuntimeError(f"git fetch pull/{other_num}/head failed: {fetch.stderr}")

    merge = _run_git(
        repo_root,
        [
            "-c",
            "user.name=ticktack-ci",
            "-c",
            "user.email=ci@ticktack.local",
            "merge",
            "--no-commit",
            "--no-ff",
            "FETCH_HEAD",
        ],
    )
    if merge.returncode != 0:
        print(f"PR #{other_num}: textual merge conflict (skipped tests; git already blocks silent merge).")
        print(merge.stdout.strip())
        _run_git(repo_root, ["merge", "--abort"])
        _run_git(repo_root, ["reset", "--hard", head_sha])
        _run_git(repo_root, ["clean", "-fd"])
        return TrialOutcome(other_num, title, overlap, "conflict", "skipped")

    ruff_proc = subprocess.run(["ruff", "check", "src", "tests"], cwd=repo_root, check=False)
    pytest_proc = subprocess.run(["pytest", "tests/"], cwd=repo_root, check=False)
    _run_git(repo_root, ["reset", "--hard", head_sha])
    _run_git(repo_root, ["clean", "-fd"])

    ok = ruff_proc.returncode == 0 and pytest_proc.returncode == 0
    status = "pass" if ok else f"FAIL (ruff={ruff_proc.returncode}, pytest={pytest_proc.returncode})"
    return TrialOutcome(other_num, title, overlap, "clean", status)


def print_summary(
    outcomes: list[TrialOutcome],
    skipped_other_base: list[dict[str, object]],
    base_ref: str,
) -> None:
    """Print a concise summary table of trial-merge results."""
    print("\n=== Trial Merge Summary ===")
    if skipped_other_base:
        skipped_desc = ", ".join(
            f"#{pr['number']} (base: {pr.get('baseRefName')})" for pr in skipped_other_base
        )
        print(f"Excluded PRs targeting other branches (not '{base_ref}'): {skipped_desc}")
    if not outcomes:
        print(f"No open PRs targeting '{base_ref}' overlap files with this PR.")
        return
    print("| PR | Merge | Checks | Overlapping files |")
    print("|---|---|---|---|")
    for res in outcomes:
        files_str = ", ".join(res.overlap[:5])
        if len(res.overlap) > 5:
            files_str += f" (+{len(res.overlap) - 5} more)"
        print(f"| #{res.pr_number} | {res.merge_status} | {res.test_status} | {files_str} |")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for overlapping-PR trial merges."""
    parser = argparse.ArgumentParser(description="Trial-merge overlapping open PRs and run checks.")
    parser.add_argument("--repo", required=True, help="owner/repo slug, e.g. SharmaLlama/ticktack")
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    head_sha = _run_git(args.repo_root, ["rev-parse", "HEAD"]).stdout.strip()
    current_files = current_pr_changed_files(args.repo_root, args.base_ref)
    open_prs = list_open_prs(args.repo)
    overlapping, skipped = select_overlapping_prs(open_prs, args.pr_number, args.base_ref, current_files)

    outcomes: list[TrialOutcome] = []
    for pr, overlap in overlapping:
        outcomes.append(_trial_merge_one(args.repo_root, head_sha, pr, overlap))

    print_summary(outcomes, skipped, args.base_ref)
    failed = [o for o in outcomes if o.merge_status == "clean" and o.test_status != "pass"]
    if failed:
        nums = ", ".join(f"#{o.pr_number}" for o in failed)
        print(f"ERROR: Silent semantic clash detected with overlapping PR(s): {nums}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
