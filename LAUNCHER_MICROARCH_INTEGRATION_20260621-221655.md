# LAUNCHER_MICROARCH_INTEGRATION_20260621-221655.md

## Context (user-provided)
- Rebase of harrypm/tape-decode-rust onto namazso/tape-decode-rs was done.
- GUI microarch selector (Auto/v1/v2/v3/v4) + decode_runtime.py helpers (resolve_tape_decode_prefix(level), build_cargo_command, cargo_env_for_level, target-x86-64-vN layout) were implemented and present in the tree.
- User states the rebase/integration "ignored the GUI update/integration that was key".
- The requirement is that released decode-light (launcher GUI) artifacts must reflect the version profile / microarch level switching capability.

## Diagnosis (file inspection + workflow reads)
Main CLI build:
- .github/workflows/build.yml
  - Windows/Linux/macOS CLI jobs build v1-v4 (and aarch64 on linux/macos).
  - Current package steps do `cp .../tape-decode .../tape-decode-${LEVEL}/tape-decode` using the same source path for every LEVEL.
  - No CARGO_TARGET_DIR is set in the build steps; all builds go into the same target/... and overwrite each other.
  - Result: the released per-level directories contain identical binaries (not distinct optimized builds). The release body claims the feature.

Launcher-specific CI (separate workflows):
- build_linux_decode.yml
  - Matrix: x86_64 (ubuntu-latest) and arm64 (ubuntu-24.04-arm).
  - Single `cargo build --release --target ${{ matrix.target }} --bin tape-decode`.
  - TAPE_DECODE_BIN: target/${{ matrix.target }}/release/tape-decode
  - Then `python3 scripts/ci/build-linux-decode-bin.py`.
  - Final artifacts: linux_decode-light_*_{x86_64,arm64}.AppImage wrapped in zips. One binary per artifact.

- build_macos_decode.yml
  - Matrix: arm64 (macos-latest) and x86_64 (macos-15-intel).
  - Same single-build + TAPE_DECODE_BIN pattern.
  - Final artifacts: macos_decode-light_*_{arm64,x86_64}.dmg.

- build_windows_decode.yml
  - Matrix: x86_64 (windows-2022) and arm64 (windows-11-arm).
  - Same single-build + TAPE_DECODE_BIN pattern (with .exe).
  - Final artifacts: windows_decode-light_*_{x86_64,arm64}.zip (renamed exe inside).

Packaging scripts (scripts/ci/):
- build-linux-decode-bin.py
  - Hardcoded candidates: TAPE_DECODE_BIN, target/x86_64-unknown-linux-gnu/release/tape-decode, target/release/tape-decode.
  - Only one --add-binary to PyInstaller for decode-light.

- build-macos-decode-bin.py
  - Hardcoded candidates for arm64 and x86_64 darwin triples + target/release.
  - Only one --add-binary.

- build-windows-decode-bin.py
  - Hardcoded candidates for x86_64-pc-windows-msvc + target/release.
  - Only one --add-binary.

Runtime/launcher source:
- decode_runtime.py
  - MICROARCH_LEVELS = ("x86-64-v1", "x86-64-v2", "x86-64-v3", "x86-64-v4")
  - _candidate_binary_paths(level) does look for:
    - meipass bundle path
    - exe_dir and nearby
    - repo_root/target-{level}/<triple>/release/tape-decode (for explicit level)
    - repo_root/target/... standard triples
  - resolve_tape_decode_prefix(level) uses the above + PATH + cargo run fallback.
- decode_launcher.py
  - MICROARCH_UI_OPTIONS built from MICROARCH_LEVELS.
  - _selected_microarch_level(), _build_command(..., level), _terminal_preview_command, _build_for_selected_level, _locate_level_binary, _selected_microarch_env all wire the level through.
  - Non-x86_64 hosts disable the controls.

Gap:
- Launcher packaging never builds multiple levels for x86_64 runners.
- Packaging scripts never discover or bundle multiple target-x86-64-vN binaries.
- Auto discovery in the bundled decode-light will only find the single binary that was bundled (unless extra target-x86-64-vN dirs are present inside the bundle or next to the exe, which they are not).
- Even in the main CLI build, the per-level directories don't contain distinct optimized binaries because CARGO_TARGET_DIR isn't used and copies come from the same path.

## Decisions
- Keep the launcher GUI and runtime code as-is (it is already complete for the end-user local use case with a source tree).
- Make the released decode-light (GUI) artifacts actually ship multiple optimized binaries for x86_64 so the selector can pick them.
- Strategy:
  1. Enhance decode_runtime.py Auto discovery to also scan target-x86-64-vN dirs highest-first when no explicit level is requested; this makes Auto prefer a level build if present.
  2. Update the three launcher packaging scripts to discover and bundle multiple levels under distinct names (e.g., tape-decode.x86-64-vN) while preserving the legacy single-binary path as a fallback.
  3. Update the three launcher CI workflows so that x86_64 matrix rows build v1-v4 using CARGO_TARGET_DIR=target-x86-64-vN + appropriate RUSTFLAGS. Arm64 rows remain single native builds.
  4. Align the main CLI build.yml to set CARGO_TARGET_DIR per level and copy from the correct per-target-dir paths so the published zips actually contain different binaries.
- Arm64 builds remain single-binary (no microarch levels apply).
- Keep TAPE_DECODE_BIN override for power users / local packaging runs.
- Local validation will be compile/import/cargo checks only. Real CI runs are required for artifact confirmation.

## Files to change (plan)
- decode_runtime.py (Auto discovery scan order)
- scripts/ci/build-linux-decode-bin.py
- scripts/ci/build-macos-decode-bin.py
- scripts/ci/build-windows-decode-bin.py
- .github/workflows/build_linux_decode.yml (x86_64 row)
- .github/workflows/build_macos_decode.yml (x86_64 row)
- .github/workflows/build_windows_decode.yml (x86_64 row)
- .github/workflows/build.yml (CLI build + package steps for per-level dirs)

## Commands run in this session
- git log --oneline --all -- .github/workflows/build.yml | head -15
- read main build.yml, three launcher ymls, three packaging scripts
- read decode.py, relevant decode_launcher.py and decode_runtime.py sections
- grep for resolve_tape_decode, MICROARCH, target-x86-64-v, CARGO_TARGET_DIR
- python3 -c 'import decode_runtime as dr; ...' to print LEVELS and helpers

## Current tree state (summary)
- GUI selector + runtime helpers: present and wired.
- Launcher artifacts: still single-binary; no level matrix in launcher CI.
- Main CLI artifacts: per-level dirs exist in layout, but contents are identical due to missing CARGO_TARGET_DIR + same-source cp.

## Next verification (user action required)
After edits and commit/push:
- Tag or manually dispatch the three launcher workflows targeting x86_64 runners.
- Inspect the produced x86_64 decode-light artifact:
  - Linux AppImage or zip should contain (inside or next to decode-light) multiple tape-decode.* or under a resources layout reflecting v1-v4.
  - macOS DMG similarly.
  - Windows zip similarly.
- Launch the GUI from the artifact on an x86_64 host and use "Locate binary" + "Build level..." to confirm the selector picks distinct binaries for each level.
- The main CLI zips should also contain distinct optimized binaries under tape-decode-{vN}/.

Timestamp: 2026-06-21T22:16:55Z
