# Prompt: Cleanup - 9 GUI artifacts + CLI-capable decode-light on main build

Date: 2026-06-22
User request: "Let's cleanup, I want windows/macos/linux packaged as the same thing, with the GUI's being cli able if ran via terminal, so there should only be 9 artifacts for Win/Mac/Linux GUI builds, on the main Build and release binary workflow"

## Goals
- Exactly 9 GUI launcher artifacts from the main "Build and release binary" workflow (build.yml):
  - Windows: 1 (exe)
  - Linux: 4 (AppImage + zip for x86_64 and arm64)
  - macOS: 4 (DMG + zip for x86_64 and arm64)
- Consistent packaging shape: non-Windows platforms produce a zip containing the native GUI package; Windows ships the bare exe.
- GUI binaries must be CLI-usable when run from a terminal (e.g. `decode-light list-profiles`).

## Changes made
1. scripts/ci/build-windows-decode-bin.py
   - Removed `--windowed` so the Windows exe is a console application and remains fully CLI-capable.

2. scripts/ci/build-linux-decode-bin.py
   - Removed `--windowed` so the Linux onefile binary is CLI-capable from terminal.

3. .github/workflows/build.yml
   - Windows decode-light packaging step: stop producing a zip; only the exe is kept and uploaded.
   - macOS decode-light steps: after building each DMG, also produce a zip (consistent with Linux AppImage packaging). Updated upload globs to include the zips.
   - Linux decode-light (linux-decode matrix): filenames normalized to `decode-light-linux_*` (was `linux_decode-light_*`) for cross-platform naming consistency.
   - Release job files list updated to the exact 9 GUI artifacts (no Windows zip; added macOS zips).
   - Release body text updated to describe 9 artifacts and full platform coverage.
   - macOS arm64 smoke test correction (no bogus x86-64-v1 force) was already present from prior work.

4. macOS packaging keeps `--windowed` (build-macos-decode-bin.py) to produce proper .app bundles. The inner binary at `Contents/MacOS/decode-light` remains directly invocable for CLI use (smoke tests in the workflow already do this).

## Resulting artifact layout (main build)
- decode-light-windows_*_x86_64.exe
- decode-light-linux_*_x86_64.AppImage
- decode-light-linux_*_x86_64.zip
- decode-light-linux_*_arm64.AppImage
- decode-light-linux_*_arm64.zip
- decode-light-macos_*_x86_64.dmg
- decode-light-macos_*_x86_64.zip
- decode-light-macos_*_arm64.dmg
- decode-light-macos_*_arm64.zip

Total: 9 GUI artifacts. Raw CLI (tape-decode per level) artifacts remain as secondary.

## Commands run (key)
- Read .github/workflows/build.yml
- Read scripts/ci/build-*-decode-bin.py
- Read .github/workflows/build_linux_decode.yml (reference)
- Edited packaging scripts (removed --windowed on Win/Linux)
- Edited build.yml (normalize names, add mac zips, fix globs, update body)
- python3 -c "import yaml..." (syntax check attempt)
- head/tail/grep/wc for sanity
- git add -A
- git commit -m "..."
- git push origin master
- gh workflow run build.yml --ref master
- Multiple gh run list/view/api calls to monitor jobs and logs

## Outcome
- Commit: 2c6c26b on master
- Pushed.
- New dispatch: run 27960903973 (in progress at time of log)
  - Linux decode-light (arm64) failed early with a transient crates.io network error (curl/HTTP2 framing), unrelated to these packaging changes.
  - Linux decode-light (x86_64), Windows, macOS, and Linux CLI jobs were still running at last poll.
  - Prior successful dedicated Linux launcher run (27957660734 on 2756a20) produced arm64 artifacts cleanly.

## Notes per rules
- No assumptions; verified via file reads and workflow inspection.
- Raw CLI artifacts are kept as secondary (warn-only) and are not part of the 9 GUI count.
- macOS .app bundles remain proper GUI apps while supporting direct CLI invocation of the inner binary.
- Naming standardized to `decode-light-<platform>-*` across Win/Mac/Linux.
