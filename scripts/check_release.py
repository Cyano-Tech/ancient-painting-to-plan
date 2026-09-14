"""Check the release file set for common credential/path leaks and broken docs links.

This bounded heuristic is not a substitute for human review or a secret scanner.
It never prints matching credential values, only the file and check name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "artifacts", "runs", "build", "dist", ".pytest_cache", ".ruff_cache", "__pycache__"}
TEXT_TYPES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".html", ".svg", ".lock", ".txt"}
PROHIBITED_SUFFIXES = {".pt", ".pth", ".safetensors", ".env", ".pem", ".key", ".pyc"}


def files_for_check(root: Path) -> list[Path]:
    if (root / ".git").is_dir():
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        if any(tracked):
            return [root / p for p in tracked if p]
    return [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not any(
            part in SKIP or part.startswith(".venv") or part.endswith(".egg-info")
            for part in p.relative_to(root).parts
        )
    ]


def inspect(root: Path) -> list[dict]:
    issues = []
    # Split fixed strings so the scanner does not trigger on its own source.
    patterns = {
        "private_home_path": re.compile("/" + "home" + r"/[A-Za-z0-9_\-]+/"),
        "workspace_endpoint": re.compile(r"https://w" + r"s-[a-z0-9]+\."),
        "api_key_shape": re.compile(r"\bs" + r"k-(?:ws-)?[A-Za-z0-9._-]{18,}"),
        "github_token_shape": re.compile(r"\bgh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    }
    for p in files_for_check(root):
        relative = str(p.relative_to(root))
        if p.is_symlink():
            issues.append({"file": relative, "check": "symlink"})
            continue
        if p.suffix in PROHIBITED_SUFFIXES or p.name == "config.local.json":
            issues.append({"file": relative, "check": "private_or_model_file"})
        if p.stat().st_size > 10 * 1024 * 1024:
            issues.append({"file": relative, "check": "file_exceeds_10_MiB"})
        if p.suffix not in TEXT_TYPES:
            continue
        content = p.read_text(encoding="utf-8")
        for label, regex in patterns.items():
            if regex.search(content):
                issues.append({"file": relative, "check": label})
        if p.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
                target = target.strip("<>")
                if urlsplit(target).scheme or target.startswith("#"):
                    continue
                destination = (p.parent / unquote(target.split("#")[0])).resolve()
                if not destination.is_relative_to(root.resolve()) or not destination.exists():
                    issues.append(
                        {
                            "file": relative,
                            "check": "missing_or_external_local_link",
                            "target": target,
                        }
                    )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    issues = inspect(args.root.resolve())
    print(
        json.dumps(
            {"issues": issues, "checked_files": len(files_for_check(args.root.resolve()))}, indent=2
        )
    )
    return bool(issues)


if __name__ == "__main__":
    raise SystemExit(main())
