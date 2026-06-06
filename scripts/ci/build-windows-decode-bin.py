#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import PyInstaller.__main__


os.environ.setdefault("SETUPTOOLS_RUST_CARGO_PROFILE", "release")


def _resolve_tape_decode_bin() -> Path:
    candidates = [
        Path(os.environ.get("TAPE_DECODE_BIN", "")),
        Path(r"target\x86_64-pc-windows-msvc\release\tape-decode.exe"),
        Path(r"target\release\tape-decode.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find tape-decode.exe. Build it before running this packaging script."
    )


def main() -> None:
    tape_decode_bin = _resolve_tape_decode_bin()
    print(f"Bundling {tape_decode_bin}")

    PyInstaller.__main__.run(
        [
            "decode.py",
            "--collect-all",
            "PyQt6",
            "--hidden-import",
            "decode_launcher",
            "--hidden-import",
            "decode_runtime",
            "--add-binary",
            f"{tape_decode_bin};.",
            "--onefile",
            "--windowed",
            "--name",
            "decode-light",
        ]
    )


if __name__ == "__main__":
    main()
