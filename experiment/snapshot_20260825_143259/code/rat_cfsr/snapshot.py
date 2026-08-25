from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "GlobecomPOWDER",
    "S3R",
    "tmp",
    "outputs",
    "logs",
    "experiment",
}


def _should_exclude(path: Path) -> bool:
    parts = path.parts
    for excluded in DEFAULT_EXCLUDES:
        excluded_parts = Path(excluded).parts
        if parts[: len(excluded_parts)] == excluded_parts:
            return True
    return False


def _git_description(repo_root: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "git: unavailable"
    return f"git_head: {head}; dirty: {bool(status)}"


def copy_current_snapshot(
    repo_root: Path,
    output_root: Path,
    snapshot_root: Path = Path("experiment"),
) -> Path:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    snapshot_root = (repo_root / snapshot_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = snapshot_root / f"snapshot_{timestamp}"
    suffix = 1
    while snapshot_dir.exists():
        snapshot_dir = snapshot_root / f"snapshot_{timestamp}_{suffix}"
        suffix += 1

    code_dir = snapshot_dir / "code"
    results_dir = snapshot_dir / "outputs" / output_root.name
    snapshot_dir.mkdir(parents=True)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set()
        directory_path = Path(directory).resolve()
        for name in names:
            rel = (directory_path / name).relative_to(repo_root)
            if _should_exclude(rel):
                ignored.add(name)
        return ignored

    shutil.copytree(repo_root, code_dir, ignore=ignore)
    if output_root.exists():
        shutil.copytree(output_root, results_dir)

    manifest = [
        f"snapshot: {snapshot_dir}",
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"code_source: {_git_description(repo_root)}",
        f"results_source: {output_root if output_root.exists() else 'missing'}",
        "contents:",
        "- code/: current project code excluding data, logs, outputs, experiments, and git metadata",
        f"- outputs/{output_root.name}/: current experiment results",
        "",
    ]
    (snapshot_dir / "SNAPSHOT.txt").write_text("\n".join(manifest), encoding="utf-8")
    return snapshot_dir
