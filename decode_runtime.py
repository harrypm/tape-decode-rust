#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


# x86-64 microarchitecture levels, ordered lowest to highest. These map directly
# to the Rust `-C target-cpu` values for x86_64, and to the artifact directory
# names emitted by the project's CI workflow (.github/workflows/build.yml):
#     CARGO_TARGET_DIR=target-x86-64-vN        cargo build --release ...
# Mirroring the CI convention lets locally-built artifacts be picked up
# without any extra wiring.
MICROARCH_LEVELS: tuple[str, ...] = ("x86-64-v1", "x86-64-v2", "x86-64-v3", "x86-64-v4")
MICROARCH_AUTO: str = ""
MICROARCH_LEVEL_TO_TARGET_CPU: dict[str, str] = {
    "x86-64-v1": "x86-64",
    "x86-64-v2": "x86-64-v2",
    "x86-64-v3": "x86-64-v3",
    "x86-64-v4": "x86-64-v4",
}


def _binary_name() -> str:
    return "tape-decode.exe" if os.name == "nt" else "tape-decode"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _native_triple() -> str:
    """Return the Rust target triple that matches the current OS/arch.

    Used for the `--target` argument when invoking cargo locally.
    """
    system = sys.platform
    machine = platform.machine().lower()
    if system.startswith("linux"):
        if machine in ("x86_64", "amd64"):
            return "x86_64-unknown-linux-gnu"
        if machine in ("aarch64", "arm64"):
            return "aarch64-unknown-linux-gnu"
    if system == "darwin":
        if machine in ("x86_64", "amd64"):
            return "x86_64-apple-darwin"
        if machine in ("aarch64", "arm64"):
            return "aarch64-apple-darwin"
    if os.name == "nt":
        if machine in ("x86_64", "amd64"):
            return "x86_64-pc-windows-msvc"
        if machine in ("aarch64", "arm64"):
            return "aarch64-pc-windows-msvc"
    raise RuntimeError(f"Unsupported host platform: {system}/{machine}")


def native_host_arch() -> str:
    """Return 'x86_64' / 'aarch64' / 'unknown' for the current host."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return "unknown"


def normalize_microarch_level(level: Optional[str]) -> str:
    """Coerce arbitrary user input to a known level or MICROARCH_AUTO."""
    if not level:
        return MICROARCH_AUTO
    candidate = level.strip().lower()
    for entry in MICROARCH_LEVELS:
        if candidate == entry or candidate == entry.replace("x86-64-", ""):
            return entry
    return MICROARCH_AUTO


def microarch_target_cpu(level: str) -> str:
    """Map a microarch level to the corresponding `-C target-cpu` value."""
    return MICROARCH_LEVEL_TO_TARGET_CPU.get(level, "native")


def microarch_target_dir(level: str) -> Optional[str]:
    """Return the CARGO_TARGET_DIR matching the level, or None for Auto."""
    if not level:
        return None
    return f"target-{level}"


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


def _candidate_binary_paths(level: str = MICROARCH_AUTO) -> list[Path]:
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

    try:
        triple = _native_triple()
    except RuntimeError:
        triple = None
    # When a microarch level is requested, look in the matching per-level
    # `target-<level>/` subdirectory first -- this is the layout produced by
    # the project's CI workflow (CARGO_TARGET_DIR=target-x86-64-vN).

    if level and triple:
        candidates.append(repo_root / f"target-{level}" / triple / "release" / binary)
    # Auto should still use the bundled/local optimized binaries when present.
    # Search highest to lowest so a released launcher with v1-v4 payloads uses
    # the best available x86-64 build unless the user picks a specific level.
    if not level and triple:
        for auto_level in reversed(MICROARCH_LEVELS):
            candidates.append(
                repo_root / f"target-{auto_level}" / triple / "release" / binary
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


def resolve_tape_decode_prefix(level: str = MICROARCH_AUTO) -> list[str]:
    level = normalize_microarch_level(level)
    for candidate in _candidate_binary_paths(level):
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


def build_tape_decode_command(
    args: Sequence[str],
    *,
    level: str = MICROARCH_AUTO,
) -> list[str]:
    return resolve_tape_decode_prefix(level=level) + list(args)


def build_cargo_command(level: str) -> list[str]:
    """Return the `cargo build` command line for rebuilding `tape-decode`
    at the given microarchitecture level. Uses the host triple and the
    `target-x86-64-vN` directory layout matching the project's CI.
    """
    level = normalize_microarch_level(level)
    if not level:
        raise ValueError("An explicit microarch level is required.")
    triple = _native_triple()
    cmd = [
        "cargo",
        "build",
        "--release",
        "--target",
        triple,
        "--bin",
        "tape-decode",
    ]
    return cmd


def cargo_env_for_level(level: str) -> dict[str, str]:
    """Environment additions needed for a level-specific cargo build."""
    level = normalize_microarch_level(level)
    env = os.environ.copy()
    if not level:
        return env
    target_cpu = microarch_target_cpu(level)
    rustflags = env.get("RUSTFLAGS", "").strip()
    flag = f"-C target-cpu={target_cpu}"
    env["RUSTFLAGS"] = f"{rustflags} {flag}".strip() if rustflags else flag
    env["CARGO_TARGET_DIR"] = microarch_target_dir(level) or env.get(
        "CARGO_TARGET_DIR", "target"
    )
    return env


def run_tape_decode(
    args: Sequence[str],
    *,
    cwd: str | None = None,
    level: str = MICROARCH_AUTO,
) -> int:
    command = build_tape_decode_command(args, level=level)
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
