#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import PyInstaller.__main__

os.environ.setdefault("SETUPTOOLS_RUST_CARGO_PROFILE", "release")


_LEVELS: tuple[str, ...] = ("x86-64-v1", "x86-64-v2", "x86-64-v3", "x86-64-v4")
# Triples we may produce level builds for on Linux CI
_TRIPLES: tuple[str, ...] = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu")


def _binary_name() -> str:
    return "tape-decode.exe" if os.name == "nt" else "tape-decode"


def _platform_sep() -> str:
    return ";" if os.name == "nt" else ":"


def _resolve_tape_decode_bin() -> Path:
    explicit = os.environ.get("TAPE_DECODE_BIN", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p.resolve()
    bin_name = _binary_name()
    repo_root = Path.cwd()
    # Prefer a level build (lowest first = v1) as the default root binary.
    # This ensures the bare "." binary inside the bundle is always runnable
    # on any x86-64 host (CI verification and end-users on older CPUs).
    for lvl in _LEVELS:
        for tri in _TRIPLES:
            p = repo_root / f"target-{lvl}" / tri / "release" / bin_name
            if p.is_file():
                return p.resolve()
    # Legacy single-build locations and generic target/
    candidates = [
        Path("target/x86_64-unknown-linux-gnu/release/tape-decode"),
        Path("target/aarch64-unknown-linux-gnu/release/tape-decode"),
        Path("target/release/tape-decode"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find tape-decode Linux binary. Build it before running this packaging script."
    )


def _discover_level_binaries() -> list[tuple[Path, str]]:
    """Find per-level optimized binaries laid out as target-x86-64-vN/<triple>/release/...

    Returns (src_path, dest_rel) suitable for --add-binary.
    """
    bin_name = _binary_name()
    repo_root = Path.cwd()
    results: list[tuple[Path, str]] = []
    for lvl in _LEVELS:
        for tri in _TRIPLES:
            p = repo_root / f"target-{lvl}" / tri / "release" / bin_name
            if p.is_file():
                dest = f"target-{lvl}/{tri}/release/{bin_name}"
                results.append((p.resolve(), dest))
                break  # only one triple per level per job
    return results


def main() -> None:
    tape_decode_bin = _resolve_tape_decode_bin()
    print(f"Bundling default {tape_decode_bin}")

    pyi_args: list[str] = [
        "decode.py",
        "--collect-all",
        "PyQt6",
        "--hidden-import",
        "decode_launcher",
        "--hidden-import",
        "decode_runtime",
        "--add-binary",
        f"{tape_decode_bin}{_platform_sep()}.",
        "--add-data",
        f"crates/tape-decode-cli/src/profiles/profiles.json{_platform_sep()}.",
        # Bundle icon assets so _resolve_icon_path can find them inside onefile bundles
        "--add-data",
        f"resources/icon/tape-decode-rust-256.png{_platform_sep()}resources/icon/tape-decode-rust-256.png",
        "--add-data",
        f"resources/icon/tape-decode-rust-256.png{_platform_sep()}tape-decode-rust-256.png",
        "--add-data",
        f"resources/icon/tape-decode-rust-256.png{_platform_sep()}decode-light.png",
        "--icon",
        "resources/icon/tape-decode-rust-256.png",
        "--onefile",
        "--windowed",
        "--name",
        "decode-light",
    ]

    for src, dest in _discover_level_binaries():
        print(f"Bundling level binary {src} -> {dest}")
        pyi_args += ["--add-binary", f"{src}{_platform_sep()}{dest}"]

    PyInstaller.__main__.run(pyi_args)


if __name__ == "__main__":
    main()
