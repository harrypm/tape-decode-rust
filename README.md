# tape-decode

A decoder for analog tape formats, written in Rust. Ported from the [vhs-decode](https://github.com/oyvindln/vhs-decode) project, commit [fe3f6099](https://github.com/oyvindln/vhs-decode/commit/fe3f6099e9e6a77295f26585598f658f2d926bb4).

## Installation

### From source

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```


### Pre-built binaries

Pre-built binaries for x64 Windows and Linux (glibc) are available in Releases. Ensure you use the correct one for your CPU feature level. On aarch64 it is recommended that you always build from source for the native architecture, as extension support is more varied.

Cross-platform GUI package workflows are also available for:
- Windows launcher EXE (x86_64 + arm64): `.github/workflows/build_windows_decode.yml`
- macOS app bundle + DMG (x86_64 + arm64): `.github/workflows/build_macos_decode.yml`
- Linux AppImage (x86_64 + aarch64): `.github/workflows/build_linux_decode.yml`

## Usage

```bash
tape-decode --help
```

## Decode Launcher GUI (Qt6)

The repository includes a Qt6 launcher (`decode.py` + `decode_launcher.py`) modeled after the vhs-decode Decode Launcher and wired to `tape-decode`.

### Run from source

```bash
python3 -m venv .venv-launcher
source .venv-launcher/bin/activate
python -m pip install -r requirements-launcher.txt
python decode.py
```

If your distro uses an externally-managed Python environment (PEP 668), use this venv flow instead of installing launcher dependencies into system Python.

The launcher defaults to guided `tape-decode decode` command creation (profile/output/frequency/threads), and can also run `list-profiles`, `compare`, and `write-profile` in a terminal.

### CLI passthrough via dispatcher

`decode.py` also forwards normal CLI args directly to `tape-decode`:

```bash
python3 decode.py decode --profile PAL_VHS --luma-out out.tbc capture.flac
```

### Build/package notes

- Windows EXE launcher bundle: `scripts/ci/build-windows-decode-bin.py`
- macOS app bundle launcher: `scripts/ci/build-macos-decode-bin.py`
- Linux launcher binary for AppImage staging: `scripts/ci/build-linux-decode-bin.py`

Linux local packaging sequence (matching CI workflow):

```bash
cargo build --release --target x86_64-unknown-linux-gnu --bin tape-decode
source .venv-launcher/bin/activate
python -m pip install pyinstaller -r requirements-launcher.txt
TAPE_DECODE_BIN=target/x86_64-unknown-linux-gnu/release/tape-decode \
  python scripts/ci/build-linux-decode-bin.py
```

For Linux arm64 local builds, replace `x86_64-unknown-linux-gnu` with `aarch64-unknown-linux-gnu` in both commands.

For release artifacts, trigger:
- `.github/workflows/build_windows_decode.yml`
- `.github/workflows/build_macos_decode.yml`
- `.github/workflows/build_linux_decode.yml`

### Examples

**List available profiles**

```bash
tape-decode list-profiles
```

Output:

```text
405_BETAMAX
819_QUADRUPLEX
MESECAM_VHS
...
```

**Decode a 40 MHz PAL VHS tape from `capture.flac`**

```bash
tape-decode decode \
  --luma-out decoded.tbc \
  --chroma-out decoded_chroma.tbc \
  --metadata-out decoded.tbc.json \
  --profile PAL_VHS \
  --frequency 40 \
  --input-format flac \
  capture.flac
```

**Decode a 16 MHZ NTSC VHS tape from `capture.u8`, with 16 threads and 60 field per-thread offset**

```bash
tape-decode decode \
  --luma-out decoded.tbc \
  --chroma-out decoded_chroma.tbc \
  --metadata-out decoded.tbc.json \
  --profile NTSC_VHS \
  --frequency 16 \
  --mt-threads 16 \
  --mt-distance-size 60 \
  capture.u8
```

**Livestream 40 MHz PAL VHS from `/dev/cxadc0`**

```bash
cat /dev/cxadc0 \
  | tape-decode decode \
    --luma-out - \
    --profile PAL_VHS \
    --frequency 40 \
    --mt-threads 16 \
    --mt-distance-size 60 \
    - \
  | ffmpeg \
    -f rawvideo \
    -pixel_format gray16le \
    -video_size 1135x626 \
    -r 25 \
    -i - \
    -f yuv4mpegpipe \
    -filter:v "format=yuv444p" \
    - \
  | mpv -
```

## Using in your project

The tape-decode crate hosting the main decoder can be used as a library in your Rust project. You can also use a `cdylib` to call the decoder from other languages.

## License

This project is based on vhs-decode, which is licensed under GPL-3.0. The Rust port is also licensed under GPL-3.0. See [COPYING](COPYING) for details.
