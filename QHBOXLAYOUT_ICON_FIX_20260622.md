# QHBoxLayout Crash + Icon Resolution Fix Log
Date: 2026-06-22
Context: Follow-up to Windows decode-light crash after Build-level removal and reported low-res icon in artifacts.

## Reported Symptoms
- Windows decode-light: crash at startup
  ```
  File "decode_launcher.py", line 443, in _build_layout
  NameError: name 'QHBoxLayout' is not defined. Did you mean: 'QVBoxLayout'?
  ```
- Icon incorrect resolution in built artifacts (all platforms requested for verification).

## Root Cause Analysis (Hard Data)
1. QHBoxLayout usage without import:
   - `grep -n "QHBoxLayout" decode_launcher.py` returned only:
     ```
     443:        action_row = QHBoxLayout()
     ```
   - Import block (lines ~29-44) contained:
     ```
     from PyQt6.QtWidgets import (
         ...
         QVBoxLayout,
         QWidget,
     )
     ```
   - No `QHBoxLayout` in the import list. The only prior use of QHBoxLayout was the removed microarch action row; the final action_row in `_build_layout` still uses it directly.

2. Icon files on disk (pre-fix):
   - `resources/icon/tape-decode-rust.ico`: 669 bytes
   - Verified via Pillow:
     ```
     Source PNG: 328318 bytes
     Current ICO: 669 bytes
     Verified ICO sizes: [(16, 16)]
     ```
   - `.icns`: 812803 bytes (correct, multi-size)
   - `tape-decode-rust-256.png`: 42851 bytes, 256x256 (correct)

3. Packaging references (scripts/ci/*):
   - Windows: `--icon resources\\icon\\tape-decode-rust.ico`
   - macOS: `--icon resources/icon/tape-decode-rust.icns`
   - Linux: `--icon resources/icon/tape-decode-rust-256.png`
   - All three add `profiles.json` and level binaries.

## Fixes Applied (Local Working Tree)
1. decode_launcher.py:
   - Added `QHBoxLayout` to the PyQt6.QtWidgets import block.

   Diff (verified):
   ```
    from PyQt6.QtWidgets import (
        ...
   +    QHBoxLayout,
        QVBoxLayout,
        ...
    )
   ```

2. resources/icon/tape-decode-rust.ico:
   - Regenerated from source `tape-decode-rust.png` (896x897 RGBA) with Pillow.
   - Sizes: 16,24,32,48,64,128,256.
   - Result: 77282 bytes, verified embedded sizes:
     ```
     Verified sizes: [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
     ```

## Verification Steps Performed
- `python3 -m py_compile decode_launcher.py` → "Syntax OK"
- Python AST parse + string check: `QHBoxLayout` occurrences: 2 (import + usage)
- Runtime PyQt6 import test:
  ```
  from PyQt6.QtWidgets import QHBoxLayout
  h = QHBoxLayout()
  ```
  → OK
- Icon regeneration + Pillow verification (sizes present)
- Confirmed .icns and 256px PNG were already correct.

## Files Changed (Uncommitted)
- M decode_launcher.py
- M resources/icon/tape-decode-rust.ico

Git status excerpt:
```
 M decode_launcher.py
 M resources/icon/tape-decode-rust.ico
```

## Next Actions (Not Performed)
- Do not commit unless explicitly requested.
- After commit/push:
  - Re-dispatch Windows launcher workflow (to ship QHBoxLayout fix + good .ico).
  - Re-dispatch full `build.yml` (or relevant launcher jobs) to verify icon on macOS/Windows/Linux artifacts.
- On new artifacts:
  - Windows: confirm .exe has multi-size icon (Resource Hacker / sigcheck / Explorer).
  - macOS: `sips -g all Contents/Resources/*.icns` or app icon inspection.
  - Linux AppImage: check hicolor 256px and root icon.
- Confirm profile list populates (from bundled profiles.json) once GUI starts on Windows.

## Commands Run (Key Excerpts)
```bash
# Diagnosis
grep -n "QHBoxLayout" decode_launcher.py
ls -la resources/icon/
python3 -c 'from PIL import Image, os; ... print sizes ...'

# Fix import
# (edit applied via agent)

# Verify
python3 -m py_compile decode_launcher.py
python3 -c 'from PyQt6.QtWidgets import QHBoxLayout; h = QHBoxLayout()'

# Regenerate ICO (final successful)
python3 -c '
from PIL import Image
base = Image.open("resources/icon/tape-decode-rust.png").convert("RGBA")
sizes = [256,128,64,48,32,24,16]
imgs = [base.resize((s,s), Image.LANCZOS) for s in sizes]
imgs[0].save("resources/icon/tape-decode-rust.ico", format="ICO",
             sizes=[(im.width,im.height) for im in imgs],
             append_images=imgs[1:])
'
python3 -c '
from PIL import Image
ico = Image.open("resources/icon/tape-decode-rust.ico")
print(sorted(ico.info.get("sizes", [])))
'
```

## Notes
- .icns and Linux 256px PNG were already multi/high-res; only .ico was truncated.
- The crash line number (~443) matched the `action_row = QHBoxLayout()` in `_build_layout` after the Build-level removal shifted rows.
- Profile bundling (`--add-data` for profiles.json) was already present in the three CI packaging scripts from prior work.

---
This log captures inputs, commands, outputs, and hard verification. No "everything is working" claim is made; artifacts must be rebuilt and inspected on target platforms.
