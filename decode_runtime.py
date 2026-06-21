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


def _env_forced_level() -> str:
    """Check for an environment-forced microarch level (used by CI verification
    and power users). Checked before falling back to Auto behavior.
    """
    for k in ("TAPE_DECODE_MICROARCH", "MICROARCH_LEVEL", "TAPE_DECODE_LEVEL"):
        v = os.environ.get(k, "").strip()
        if v:
            return normalize_microarch_level(v)
    return MICROARCH_AUTO


def _binary_seems_runnable(p: Path, timeout_seconds: float = 4.0) -> bool:
    """Probe whether a candidate tape-decode binary can actually execute
    'list-profiles' on the current host without an illegal-instruction crash.

    Used only for Auto selection so we skip level-optimized binaries whose
    required CPU features are not available locally.
    """
    try:
        completed = subprocess.run(
            [str(p), "list-profiles"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode not in (0, 1, 2):
            return False
        combined = (completed.stdout or "") + (completed.stderr or "")
        low = combined.lower()
        if any(x in low for x in ("illegal instruction", "sigill", "trace/breakpoint")):
            return False
        return True
    except Exception:
        return False


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
    meipass_path = Path(meipass) if meipass else None

    try:
        triple = _native_triple()
    except RuntimeError:
        triple = None

    # Structured per-level paths (target-x86-64-vN/<triple>/release) are preferred
    # so that an explicit level or Auto can select the matching optimized binary
    # even when a bare default binary is also present in the bundle root.
    # For Auto we insert highest-to-lowest so the best runnable one wins.
    if triple:
        if level:
            if meipass_path:
                candidates.append(
                    meipass_path / f"target-{level}" / triple / "release" / binary
                )
            candidates.append(
                repo_root / f"target-{level}" / triple / "release" / binary
            )
        else:
            for auto_level in reversed(MICROARCH_LEVELS):
                if meipass_path:
                    candidates.append(
                        meipass_path / f"target-{auto_level}" / triple / "release" / binary
                    )
                candidates.append(
                    repo_root / f"target-{auto_level}" / triple / "release" / binary
                )

    # Bare default (the binary we place at "." inside the PyInstaller bundle).
    # This is intentionally a safe baseline (v1). It is only used if no structured
    # level binary was found or selected.
    if meipass_path:
        candidates.append(meipass_path / binary)

    exe_dir = Path(sys.executable).resolve(strict=False).parent
    candidates.extend(
        [
            exe_dir / binary,
            exe_dir / "bin" / binary,
            exe_dir / ".." / binary,
        ]
    )

    # Legacy single-target and generic layouts last.
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
    if level == MICROARCH_AUTO:
        forced = _env_forced_level()
        if forced != MICROARCH_AUTO:
            level = forced
    for candidate in _candidate_binary_paths(level):
        if candidate.is_file():
            # For explicit level we trust the on-disk match (user asked for it).
            # For Auto we only accept binaries that actually execute on this host
            # (skip higher levels whose CPU features are unavailable).
            if level != MICROARCH_AUTO or _binary_seems_runnable(candidate):
                return [str(candidate)]
            # continue to lower level for Auto
            continue

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
