#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _binary_name() -> str:
    return "tape-decode.exe" if os.name == "nt" else "tape-decode"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _candidate_binary_paths() -> list[Path]:
    binary = _binary_name()
    repo_root = _repo_root()
    candidates: list[Path] = []

    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(Path(meipass) / binary)

    exe_dir = Path(sys.executable).resolve(strict=False).parent
    candidates.extend(
        [
            exe_dir / binary,
            exe_dir / "bin" / binary,
            exe_dir / ".." / binary,
        ]
    )

    candidates.extend(
        [
            repo_root / "target" / "release" / binary,
            repo_root / "target" / "debug" / binary,
            repo_root / "target" / "x86_64-unknown-linux-gnu" / "release" / binary,
            repo_root / "target" / "x86_64-pc-windows-msvc" / "release" / binary,
            repo_root / "target" / "aarch64-apple-darwin" / "release" / binary,
            repo_root / "target" / "x86_64-apple-darwin" / "release" / binary,
        ]
    )

    return _dedupe_paths(candidates)


def resolve_tape_decode_prefix() -> list[str]:
    for candidate in _candidate_binary_paths():
        if candidate.is_file():
            return [str(candidate)]

    on_path = shutil.which(_binary_name()) or shutil.which("tape-decode")
    if on_path:
        return [on_path]

    repo_root = _repo_root()
    if (repo_root / "Cargo.toml").is_file() and shutil.which("cargo"):
        return ["cargo", "run", "--release", "--bin", "tape-decode", "--"]

    raise FileNotFoundError(
        "Could not locate tape-decode binary. Build it first or add it to PATH."
    )


def build_tape_decode_command(args: Sequence[str]) -> list[str]:
    return resolve_tape_decode_prefix() + list(args)


def run_tape_decode(args: Sequence[str], *, cwd: str | None = None) -> int:
    command = build_tape_decode_command(args)
    return subprocess.call(command, cwd=cwd)


def list_profiles(timeout_seconds: int = 20) -> list[str]:
    command = build_tape_decode_command(["list-profiles"])
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or "tape-decode list-profiles failed")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def fallback_profiles_from_repo() -> list[str]:
    profiles_json = _repo_root() / "crates" / "tape-decode-cli" / "src" / "profiles" / "profiles.json"
    if not profiles_json.is_file():
        return []
    try:
        data = json.loads(profiles_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    return sorted(name for name in data.keys() if isinstance(name, str))


def load_profiles() -> list[str]:
    fallback = fallback_profiles_from_repo()
    if fallback:
        return fallback
    try:
        profiles = list_profiles()
        if profiles:
            return profiles
    except Exception:
        pass
    return []
