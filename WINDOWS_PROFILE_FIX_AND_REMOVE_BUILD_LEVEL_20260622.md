# Windows profile list + remove Build level button

Date: 2026-06-22
Task: Fix "profile list for windows is borken" and remove the non-functional "Build level" GUI element.

## Diagnosis

### Profile list on Windows
- `load_profiles()` prefers `fallback_profiles_from_repo()` which only reads `crates/.../profiles.json` relative to `__file__` parent.
- In PyInstaller onefile bundles (`decode-light`), no source tree is present → fallback returns [].
- Then it calls `list_profiles()` → `build_tape_decode_command(["list-profiles"])` (always AUTO) → `resolve_tape_decode_prefix(AUTO)`.
- AUTO tries structured `target-x86-64-vN/<triple>/...` highest-to-lowest, then bare default (v1), with a runnable probe (`_binary_seems_runnable`) for each.
- The probe + path construction worked in Linux/mac CI only because verification forced `TAPE_DECODE_MICROARCH=x86-64-v1`.
- Windows launcher workflow has no equivalent execution verification step for the built artifact.
- Result: in released Windows x86_64/arm64 decode-light, `load_profiles()` often returned [], GUI `_refresh_profiles` only showed DEFAULT_PROFILE. Appeared "broken".

### Build level button
- `microarch_build_button` + `_build_for_selected_level` + wiring + layout row only existed to open a terminal with cargo + RUSTFLAGS + CARGO_TARGET_DIR for local rebuild of a level.
- User states it "does nothing" and should be removed.
- No other code depends on the build path; `microarch_target_*`, locate, level selector, and preview remain useful.

## Changes made
1. `decode_runtime.py`
   - Extended `fallback_profiles_from_repo()` to also search `sys._MEIPASS/profiles.json` and locations next to the executable.
   - This makes profile list population work inside all decode-light bundles (Windows, Linux, mac) without exec.

2. `scripts/ci/build-*-decode-bin.py` (all three)
   - Added `--add-data` for `crates/tape-decode-cli/src/profiles/profiles.json` → `profiles.json` at bundle root (using platform separator).
   - This populates the file that the enhanced fallback looks for.

3. `decode_launcher.py`
   - Removed `microarch_build_button` creation, tooltip, layout block (the row 9 HBox), and signal connection.
   - Shifted subsequent grid rows (include_chroma and below) up by 1 to avoid empty space.
   - Removed disabling line for the build button on non-x86.
   - Deleted the entire `_build_for_selected_level` method.
   - Removed `build_cargo_command` and `cargo_env_for_level` from the decode_runtime import.
   - Updated two user-facing messages that referenced the removed "Build level…" button (in preview and locate error).
   - The x86-64 microarch level selector, "Locate binary", AUTO/probe, explicit levels, env override, and launch/preview behavior are all preserved.

## Validation performed (local)
- python -m py_compile decode_runtime.py decode_launcher.py scripts/ci/build-*-decode-bin.py
- Import of decode_launcher succeeds and widget construction does not reference removed button.

## Notes
- Windows CI verification of list-profiles on the artifact is still absent (Linux/mac have forced-v1 checks). Consider adding later for parity.
- Profile names are static data; falling back to the bundled json is correct and robust for the GUI combo.
- "list-profiles (terminal)" tool still execs the selected level binary as before.

This log preserves the state and rationale per project rules.
