"""Execute a fast subset of example notebooks in memory for CI verification."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

DEFAULT_NOTEBOOKS = (
    "notebooks/v2/docs_examples/01_single_series_774ce.ipynb",
    "notebooks/v2/docs_examples/03_nonparametric_inverse_reconstruction.ipynb",
    "notebooks/v2/docs_examples/05_flat_baseline_smc_evidence.ipynb",
)


def execute_notebook(nb_path: Path, workdir: Path, timeout: int = 300) -> float:
    """Execute one notebook in memory using ``nbclient`` and return elapsed wall seconds."""
    import nbclient
    import nbformat

    os.environ["MPLBACKEND"] = "Agg"
    notebook = nbformat.read(nb_path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
    )
    t0 = time.perf_counter()
    client.execute()
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for CI notebook execution."""
    parser = argparse.ArgumentParser(description="Execute fast ticktack example notebooks.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("notebooks", nargs="*", default=list(DEFAULT_NOTEBOOKS))
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    total_s = 0.0
    for rel in args.notebooks:
        nb_path = (repo_root / rel).resolve()
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")
        print(f"Executing {rel} ...", flush=True)
        elapsed = execute_notebook(nb_path, workdir=repo_root, timeout=args.timeout)
        total_s += elapsed
        print(f"PASS {rel} ({elapsed:.1f} s)", flush=True)
    print(f"All {len(args.notebooks)} notebooks passed in {total_s:.1f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
