#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from decode_runtime import build_tape_decode_command, load_profiles

try:
    from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStyleFactory,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PyQt6 is required for Decode Launcher.") from exc


ALIGN_TOP = Qt.AlignmentFlag.AlignTop


@dataclass(frozen=True)
class ToolSpec:
    label: str
    subcommand: str
    notes: str = ""


TOOLS = [
    ToolSpec(
        label="tape-decode decode (guided)",
        subcommand="decode",
        notes="Builds a decode command from the form fields and launches it in a terminal.",
    ),
    ToolSpec(
        label="tape-decode list-profiles (terminal)",
        subcommand="list-profiles",
        notes="Runs list-profiles; optional flags can be added in Extra arguments.",
    ),
    ToolSpec(
        label="tape-decode compare (terminal)",
        subcommand="compare",
        notes="Runs compare; provide required compare arguments in Extra arguments.",
    ),
    ToolSpec(
        label="tape-decode write-profile (terminal)",
        subcommand="write-profile",
        notes="Runs write-profile; provide required arguments in Extra arguments.",
    ),
]

INPUT_FORMATS = ["u8", "s8", "s16le", "u16le", "f32le", "flac"]
DEFAULT_PROFILE = "PAL_VHS"


def _split_user_args(extra_args: str, *, strict: bool = True) -> list[str]:
    if not extra_args.strip():
        return []
    try:
        parsed = shlex.split(extra_args, posix=os.name != "nt")
        if os.name == "nt":
            return [
                arg[1:-1]
                if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in {'"', "'"}
                else arg
                for arg in parsed
            ]
        return parsed
    except ValueError:
        if strict:
            raise
        return [extra_args]


def _arg_writes_raw_output_to_stdout(args: list[str]) -> bool:
    output_flags = {"--luma-out", "--chroma-out"}
    i = 0
    while i < len(args):
        token = args[i]
        if token in output_flags:
            if i + 1 < len(args) and args[i + 1] == "-":
                return True
            i += 2
            continue
        if any(token == f"{flag}=-" for flag in output_flags):
            return True
        i += 1
    return False


def _extract_dropped_file_path(mime_data, *, suffix_filter: Optional[set[str]] = None) -> Optional[str]:
    if mime_data is None:
        return None

    paths: list[str] = []
    if mime_data.hasUrls():
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local = url.toLocalFile().strip()
            if local:
                paths.append(local)
    elif mime_data.hasText():
        text = mime_data.text().strip()
        if text:
            local = QUrl(text).toLocalFile() if text.startswith("file:") else text
            if local:
                paths.append(local)

    for raw in paths:
        expanded = str(Path(raw).expanduser())
        if suffix_filter:
            suffix = Path(expanded).suffix.lower()
            if suffix not in suffix_filter:
                continue
        if Path(expanded).is_dir():
            continue
        return expanded
    return None


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _shell_join_windows(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def _shell_join_platform(parts: list[str]) -> str:
    return _shell_join_windows(parts) if os.name == "nt" else _shell_join(parts)


def _open_linux_terminal(shell_command: str) -> None:
    shell = os.environ.get("SHELL", "/bin/bash")
    shell_args = [shell, "-lc", shell_command]
    terminal_candidates: list[tuple[str, list[str]]] = [
        ("x-terminal-emulator", ["x-terminal-emulator", "-e", *shell_args]),
        ("gnome-terminal", ["gnome-terminal", "--", *shell_args]),
        ("kgx", ["kgx", "--", *shell_args]),
        ("konsole", ["konsole", "-e", *shell_args]),
        ("mate-terminal", ["mate-terminal", "--", *shell_args]),
        ("xfce4-terminal", ["xfce4-terminal", "--command", f"{shell} -lc {shlex.quote(shell_command)}"]),
        ("lxterminal", ["lxterminal", "-e", f"{shell} -lc {shlex.quote(shell_command)}"]),
        ("kitty", ["kitty", "--hold", *shell_args]),
        ("alacritty", ["alacritty", "-e", *shell_args]),
        ("xterm", ["xterm", "-hold", "-e", *shell_args]),
    ]

    for binary, command in terminal_candidates:
        if shutil.which(binary):
            subprocess.Popen(command)
            return

    raise RuntimeError(
        "Could not find a supported Linux terminal emulator (e.g. gnome-terminal, konsole, xterm)."
    )


def _open_terminal(command_parts: list[str], working_directory: Path) -> None:
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            command_parts,
            cwd=str(working_directory),
            creationflags=creation_flags,
        )
        return

    command = _shell_join(command_parts)
    shell = os.environ.get("SHELL", "/bin/bash")
    shell_command = (
        f'echo "[decode-launcher] starting command..."; '
        f"cd {shlex.quote(str(working_directory))} && {command}; "
        "status=$?; "
        "echo; "
        'echo "[decode-launcher] process finished with exit code $status"; '
        f"exec {shlex.quote(shell)} -l"
    )

    if sys.platform == "darwin":
        escaped = shell_command.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'tell application "Terminal" to activate',
                "-e",
                f'tell application "Terminal" to do script "{escaped}"',
            ]
        )
        return

    _open_linux_terminal(shell_command)


class FileDropLineEdit(QLineEdit):
    fileDropped = pyqtSignal(str)

    def __init__(self, *, suffix_filter: Optional[set[str]] = None):
        super().__init__()
        self._suffix_filter = suffix_filter
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        dropped = _extract_dropped_file_path(
            event.mimeData(), suffix_filter=self._suffix_filter
        )
        if dropped:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        dropped = _extract_dropped_file_path(
            event.mimeData(), suffix_filter=self._suffix_filter
        )
        if dropped:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        dropped = _extract_dropped_file_path(
            event.mimeData(), suffix_filter=self._suffix_filter
        )
        if dropped:
            self.setText(dropped)
            self.fileDropped.emit(dropped)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class DecodeLauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._tools = TOOLS
        self.setWindowTitle("Decode Launcher")
        self.resize(860, 360)

        self.tool_combo = QComboBox()
        for tool in self._tools:
            self.tool_combo.addItem(tool.label)
        self.tool_combo.setCurrentIndex(0)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)

        self.input_edit = FileDropLineEdit()
        self.input_edit.setPlaceholderText("Drop RF input file here")
        self.input_browse_button = QPushButton("Input…")

        self.output_edit = QLineEdit("")
        self.output_browse_button = QPushButton("Output…")

        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(True)
        self.profile_combo.addItem(DEFAULT_PROFILE)
        self.refresh_profiles_button = QPushButton("Refresh profiles")

        self.use_profile_file_check = QCheckBox("Use profile JSON file")
        self.profile_file_edit = FileDropLineEdit(suffix_filter={".json"})
        self.profile_file_edit.setPlaceholderText("Drop profile JSON file here")
        self.profile_file_browse_button = QPushButton("Profile JSON…")

        self.frequency_edit = QLineEdit("40")
        self.input_format_combo = QComboBox()
        self.input_format_combo.addItems(INPUT_FORMATS)
        self.input_format_combo.setCurrentText("flac")

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setValue(4)
        self.mt_distance_size_spin = QSpinBox()
        self.mt_distance_size_spin.setRange(1, 1000000)
        self.mt_distance_size_spin.setValue(60)

        self.include_chroma_check = QCheckBox("Write chroma output (_chroma.tbc)")
        self.include_chroma_check.setChecked(True)
        self.include_metadata_check = QCheckBox("Write metadata output (.tbc.json)")
        self.include_metadata_check.setChecked(True)
        self.ire0_adjust_check = QCheckBox("Adjust RF IRE0 (--ire0-adjust)")
        self.ire0_adjust_check.setChecked(True)

        self.overwrite_check = QCheckBox("Allow overwrite (--overwrite)")
        self.debug_check = QCheckBox("Enable debug logging (--debug)")

        self.extra_args_edit = QLineEdit("")
        self.command_preview = QLineEdit("")
        self.command_preview.setReadOnly(True)

        self.launch_button = QPushButton("Launch selected tool")
        self.launch_tbc_tools_button = QPushButton("Launch tbc-tools / ld-analyse")
        self.close_button = QPushButton("Close")

        self._output_manually_set = False

        self._build_layout()
        self._wire_events()
        self._refresh_tool_state()

        QTimer.singleShot(0, self._refresh_profiles)

    def _build_layout(self) -> None:
        root = QVBoxLayout()
        root.setAlignment(ALIGN_TOP)

        launch_group = QGroupBox("")
        launch_layout = QGridLayout()
        launch_group.setLayout(launch_layout)

        launch_layout.addWidget(QLabel("Tool"), 0, 0)
        launch_layout.addWidget(self.tool_combo, 0, 1, 1, 3)

        launch_layout.addWidget(QLabel("Input file"), 1, 0)
        launch_layout.addWidget(self.input_edit, 1, 1, 1, 2)
        launch_layout.addWidget(self.input_browse_button, 1, 3)

        launch_layout.addWidget(QLabel("Output base"), 2, 0)
        launch_layout.addWidget(self.output_edit, 2, 1, 1, 2)
        launch_layout.addWidget(self.output_browse_button, 2, 3)

        launch_layout.addWidget(QLabel("Profile"), 3, 0)
        launch_layout.addWidget(self.profile_combo, 3, 1, 1, 2)
        launch_layout.addWidget(self.refresh_profiles_button, 3, 3)

        launch_layout.addWidget(self.use_profile_file_check, 4, 0, 1, 4)

        launch_layout.addWidget(QLabel("Profile file"), 5, 0)
        launch_layout.addWidget(self.profile_file_edit, 5, 1, 1, 2)
        launch_layout.addWidget(self.profile_file_browse_button, 5, 3)

        launch_layout.addWidget(QLabel("Frequency (MHz)"), 6, 0)
        launch_layout.addWidget(self.frequency_edit, 6, 1)
        launch_layout.addWidget(QLabel("Input format"), 6, 2)
        launch_layout.addWidget(self.input_format_combo, 6, 3)

        launch_layout.addWidget(QLabel("Threads (0 = serial)"), 7, 0)
        launch_layout.addWidget(self.threads_spin, 7, 1)
        launch_layout.addWidget(QLabel("MT distance size"), 7, 2)
        launch_layout.addWidget(self.mt_distance_size_spin, 7, 3)

        launch_layout.addWidget(self.include_chroma_check, 8, 0, 1, 2)
        launch_layout.addWidget(self.include_metadata_check, 8, 2, 1, 2)
        launch_layout.addWidget(self.overwrite_check, 9, 0, 1, 2)
        launch_layout.addWidget(self.ire0_adjust_check, 9, 2, 1, 2)
        launch_layout.addWidget(self.debug_check, 10, 0, 1, 2)

        launch_layout.addWidget(QLabel("Extra arguments"), 11, 0)
        launch_layout.addWidget(self.extra_args_edit, 11, 1, 1, 3)

        launch_layout.addWidget(QLabel("Terminal preview"), 12, 0)
        launch_layout.addWidget(self.command_preview, 12, 1, 1, 3)

        launch_layout.addWidget(self.note_label, 13, 0, 1, 4)

        action_row = QHBoxLayout()
        action_row.addWidget(self.launch_button)
        action_row.addWidget(self.launch_tbc_tools_button)
        action_row.addWidget(self.close_button)

        root.addWidget(launch_group)
        root.addLayout(action_row)
        self.setLayout(root)

    def _wire_events(self) -> None:
        self.tool_combo.currentIndexChanged.connect(self._refresh_tool_state)
        self.input_edit.textChanged.connect(self._on_input_changed)
        self.output_edit.textChanged.connect(self._refresh_tool_state)
        self.output_edit.textEdited.connect(self._on_output_edited)
        self.profile_combo.currentTextChanged.connect(self._refresh_tool_state)
        self.use_profile_file_check.toggled.connect(self._refresh_tool_state)
        self.profile_file_edit.textChanged.connect(self._refresh_tool_state)
        self.frequency_edit.textChanged.connect(self._refresh_tool_state)
        self.input_format_combo.currentIndexChanged.connect(self._refresh_tool_state)
        self.threads_spin.valueChanged.connect(self._refresh_tool_state)
        self.mt_distance_size_spin.valueChanged.connect(self._refresh_tool_state)
        self.include_chroma_check.toggled.connect(self._refresh_tool_state)
        self.include_metadata_check.toggled.connect(self._refresh_tool_state)
        self.ire0_adjust_check.toggled.connect(self._refresh_tool_state)
        self.overwrite_check.toggled.connect(self._refresh_tool_state)
        self.debug_check.toggled.connect(self._refresh_tool_state)
        self.extra_args_edit.textChanged.connect(self._refresh_tool_state)

        self.refresh_profiles_button.clicked.connect(self._refresh_profiles)
        self.input_browse_button.clicked.connect(self._browse_input_file)
        self.output_browse_button.clicked.connect(self._browse_output_path)
        self.profile_file_browse_button.clicked.connect(self._browse_profile_file)
        self.launch_button.clicked.connect(self._launch_selected_tool)
        self.launch_tbc_tools_button.clicked.connect(self._launch_tbc_tools)
        self.close_button.clicked.connect(self.close)

    def _selected_tool(self) -> ToolSpec:
        return self._tools[self.tool_combo.currentIndex()]

    def _is_decode_tool(self) -> bool:
        return self._selected_tool().subcommand == "decode"

    def _refresh_profiles(self) -> None:
        current = self.profile_combo.currentText().strip()
        profiles = load_profiles()
        if not profiles:
            if self.profile_combo.count() == 0:
                self.profile_combo.addItem(DEFAULT_PROFILE)
            if not self.profile_combo.currentText().strip():
                self.profile_combo.setCurrentText(DEFAULT_PROFILE)
            self._refresh_tool_state()
            return

        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(profiles)
        if current and current in profiles:
            self.profile_combo.setCurrentText(current)
        elif DEFAULT_PROFILE in profiles:
            self.profile_combo.setCurrentText(DEFAULT_PROFILE)
        elif profiles:
            self.profile_combo.setCurrentIndex(0)
        self.profile_combo.blockSignals(False)
        self._refresh_tool_state()

    def _refresh_tool_state(self) -> None:
        tool = self._selected_tool()
        decode_selected = tool.subcommand == "decode"
        profile_file_selected = decode_selected and self.use_profile_file_check.isChecked()

        for widget in (
            self.input_edit,
            self.input_browse_button,
            self.output_edit,
            self.output_browse_button,
            self.frequency_edit,
            self.input_format_combo,
            self.threads_spin,
            self.mt_distance_size_spin,
            self.include_chroma_check,
            self.include_metadata_check,
            self.ire0_adjust_check,
            self.overwrite_check,
            self.debug_check,
            self.use_profile_file_check,
            self.refresh_profiles_button,
        ):
            widget.setEnabled(decode_selected)

        self.profile_combo.setEnabled(decode_selected and not profile_file_selected)
        self.profile_file_edit.setEnabled(profile_file_selected)
        self.profile_file_browse_button.setEnabled(profile_file_selected)

        self.command_preview.setText(self._terminal_preview_command(tool))
        self.note_label.setText(tool.notes)

    def _infer_default_output_base(self, input_path: str) -> str:
        if not input_path.strip():
            return ""
        path = Path(input_path.strip()).expanduser()
        parent = path.parent if path.parent.as_posix() != "." else Path.cwd()
        stem = path.stem or path.name
        return str(parent / stem)

    def _on_input_changed(self, value: str) -> None:
        if not self._output_manually_set:
            self.output_edit.setText(self._infer_default_output_base(value))
        self._refresh_tool_state()

    def _on_output_edited(self, value: str) -> None:
        self._output_manually_set = bool(value.strip())

    def _derive_output_paths(self, output_base: str) -> tuple[str, str, str]:
        output_base = output_base.strip()
        if not output_base:
            return "", "", ""
        if output_base.lower().endswith(".tbc"):
            luma = output_base
            base = output_base[: -len(".tbc")]
        else:
            luma = f"{output_base}.tbc"
            base = output_base
        chroma = f"{base}_chroma.tbc"
        metadata = f"{base}.tbc.json"
        return luma, chroma, metadata

    def _build_decode_args(self, *, strict: bool) -> list[str]:
        args = ["decode"]
        input_path = self.input_edit.text().strip()
        output_base = self.output_edit.text().strip()
        luma_out, chroma_out, metadata_out = self._derive_output_paths(output_base)

        if strict:
            if not input_path:
                raise RuntimeError("Select an input file.")
            if not output_base:
                raise RuntimeError("Select an output base path.")

        if luma_out:
            args += ["--luma-out", luma_out]
        if chroma_out and self.include_chroma_check.isChecked():
            args += ["--chroma-out", chroma_out]
        if metadata_out and self.include_metadata_check.isChecked():
            args += ["--metadata-out", metadata_out]

        frequency = self.frequency_edit.text().strip()
        if frequency:
            args += ["--frequency", frequency]

        input_format = self.input_format_combo.currentText().strip()
        if input_format:
            args += ["--input-format", input_format]

        threads = self.threads_spin.value()
        if threads > 0:
            args += ["--mt-threads", str(threads)]
            args += ["--mt-distance-size", str(self.mt_distance_size_spin.value())]

        if self.overwrite_check.isChecked():
            args.append("--overwrite")
        if self.ire0_adjust_check.isChecked():
            args.append("--ire0-adjust")
        if self.debug_check.isChecked():
            args.append("--debug")

        if self.use_profile_file_check.isChecked():
            profile_file = self.profile_file_edit.text().strip()
            if strict and not profile_file:
                raise RuntimeError("Select a profile JSON file or disable profile file mode.")
            if profile_file:
                args += ["--profile-file", profile_file]
        else:
            profile = self.profile_combo.currentText().strip()
            if strict and not profile:
                raise RuntimeError("Select a profile.")
            if profile:
                args += ["--profile", profile]

        extra = self.extra_args_edit.text().strip()
        if extra:
            extra_args = _split_user_args(extra, strict=strict)
            if _arg_writes_raw_output_to_stdout(extra_args):
                raise RuntimeError(
                    "Decode Launcher cannot use --luma-out - / --chroma-out - in Extra arguments. "
                    "Use Output base file paths in the form, or run a manual shell pipeline outside launcher."
                )
            args += extra_args

        if input_path:
            args.append(input_path)

        return args

    def _build_command(self, tool: ToolSpec, *, strict: bool) -> list[str]:
        if tool.subcommand == "decode":
            return build_tape_decode_command(self._build_decode_args(strict=strict))

        extra = self.extra_args_edit.text().strip()
        extra_args = _split_user_args(extra, strict=strict) if extra else []
        return build_tape_decode_command([tool.subcommand] + extra_args)

    def _terminal_preview_command(self, tool: ToolSpec) -> str:
        try:
            command = self._build_command(tool, strict=False)
            return _shell_join_platform(command)
        except Exception as exc:
            return f"[preview unavailable] {exc}"

    def _effective_working_directory(self) -> Path:
        input_path = self.input_edit.text().strip()
        if input_path:
            candidate = Path(input_path).expanduser()
            parent = candidate.parent
            if parent.is_dir():
                return parent.resolve()
        return Path(os.getcwd()).resolve()

    def _browse_input_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select input RF file",
            self.input_edit.text().strip() or str(self._effective_working_directory()),
            "RF captures (*.flac *.lds *.u8 *.s8 *.s16le *.u16le *.f32le *.raw *.bin);;All files (*)",
        )
        if selected:
            self.input_edit.setText(selected)

    def _browse_output_path(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Select output base name",
            self.output_edit.text().strip() or str(self._effective_working_directory()),
        )
        if selected:
            self._output_manually_set = True
            self.output_edit.setText(selected)

    def _browse_profile_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select profile JSON file",
            self.profile_file_edit.text().strip() or str(self._effective_working_directory()),
            "JSON files (*.json);;All files (*)",
        )
        if selected:
            self.profile_file_edit.setText(selected)

    def _output_to_tbc_candidate(self, output_value: str) -> Optional[Path]:
        if not output_value.strip():
            return None

        output_path = Path(output_value.strip()).expanduser()
        if not output_path.is_absolute():
            output_path = self._effective_working_directory() / output_path
        output_path = output_path.resolve(strict=False)

        if output_path.suffix.lower() in {".tbc", ".lds"}:
            return output_path
        return Path(str(output_path) + ".tbc")

    def _candidate_tbc_path(self) -> Optional[Path]:
        tbc_path = self._output_to_tbc_candidate(self.output_edit.text())
        if tbc_path is None:
            return None
        if tbc_path.exists():
            return tbc_path
        if self._is_decode_tool():
            return tbc_path
        return None

    def _candidate_tbc_tool_names(self) -> list[str]:
        if os.name == "nt":
            return ["ld-analyse.exe", "tbc-analyse.exe", "tbc-tools.exe"]
        if sys.platform == "darwin":
            return ["ld-analyse", "tbc-analyse", "tbc-tools"]
        return [
            "ld-analyse",
            "tbc-analyse",
            "tbc-tools",
            "tbc-tools.AppImage",
            "tbc-tools.appimage",
            "tbc-tools-x86_64.AppImage",
            "tbc-tools-x86_64.appimage",
            "tbc-tools-aarch64.AppImage",
            "tbc-tools-aarch64.appimage",
        ]

    def _existing_parent_dir(self, raw_path: str) -> Optional[Path]:
        if not raw_path.strip():
            return None
        candidate = Path(raw_path.strip()).expanduser()
        parent = candidate.parent
        if parent.is_dir():
            return parent.resolve()
        return None

    def _candidate_tbc_search_roots(self) -> list[Path]:
        roots: list[Path] = [self._effective_working_directory()]
        input_parent = self._existing_parent_dir(self.input_edit.text())
        output_parent = self._existing_parent_dir(self.output_edit.text())
        if input_parent is not None:
            roots.append(input_parent)
        if output_parent is not None:
            roots.append(output_parent)

        if os.name == "nt":
            roots.extend(
                [
                    Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
                    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
                    Path.home() / "AppData" / "Local" / "Programs",
                ]
            )
        elif sys.platform == "darwin":
            roots.extend(
                [
                    Path("/Applications"),
                    Path.home() / "Applications",
                    Path("/opt/homebrew/bin"),
                    Path("/usr/local/bin"),
                    Path("/usr/bin"),
                ]
            )
        else:
            roots.extend(
                [
                    Path("/usr/local/bin"),
                    Path("/usr/bin"),
                    Path("/opt"),
                    Path.home() / "Applications",
                    Path.home() / "bin",
                ]
            )

        deduped: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(root)
        return deduped

    def _find_tbc_tools_executable(self) -> Optional[Path]:
        names = self._candidate_tbc_tool_names()

        for name in names:
            on_path = shutil.which(name)
            if on_path:
                return Path(on_path)

        for root in self._candidate_tbc_search_roots():
            for name in names:
                direct = root / name
                if direct.is_file():
                    return direct

                in_tbc_tools_dir = root / "tbc-tools" / name
                if in_tbc_tools_dir.is_file():
                    return in_tbc_tools_dir

            if sys.platform == "darwin":
                mac_candidates = [
                    root / "tbc-tools.app" / "Contents" / "MacOS" / "ld-analyse",
                    root / "tbc-tools.app" / "Contents" / "MacOS" / "tbc-tools",
                    root / "ld-analyse.app" / "Contents" / "MacOS" / "ld-analyse",
                ]
                for candidate in mac_candidates:
                    if candidate.is_file():
                        return candidate

        return None

    def _macos_app_bundle_for_binary(self, executable: Path) -> Optional[Path]:
        if sys.platform != "darwin":
            return None
        for parent in executable.resolve(strict=False).parents:
            if parent.suffix.lower() == ".app":
                return parent
        return None

    def _launch_tbc_tools(self) -> None:
        executable = self._find_tbc_tools_executable()
        if executable is None:
            QMessageBox.critical(
                self,
                "tbc-tools not found",
                "Could not find tbc-tools / ld-analyse in PATH or standard install locations.",
            )
            return

        tbc_candidate = self._candidate_tbc_path()
        app_bundle = self._macos_app_bundle_for_binary(executable)
        if app_bundle is not None:
            command = ["open", "-a", str(app_bundle)]
            if tbc_candidate is not None:
                command += ["--args", str(tbc_candidate)]
        else:
            command = [str(executable)]
            if tbc_candidate is not None:
                command.append(str(tbc_candidate))

        try:
            subprocess.Popen(command, cwd=str(self._effective_working_directory()))
        except Exception as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))

    def _launch_selected_tool(self) -> None:
        tool = self._selected_tool()
        working_directory = self._effective_working_directory()
        if not working_directory.is_dir():
            QMessageBox.critical(
                self,
                "Invalid working directory",
                f"Directory does not exist:\n{working_directory}",
            )
            return

        try:
            command = self._build_command(tool, strict=True)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid arguments", str(exc))
            return

        try:
            _open_terminal(command, working_directory)
        except Exception as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))


def _apply_fusion_dark_mode(app: QApplication) -> None:
    fusion_style = QStyleFactory.create("Fusion")
    if fusion_style is not None:
        app.setStyle(fusion_style)
    else:
        app.setStyle("Fusion")

    role = QPalette.ColorRole if hasattr(QPalette, "ColorRole") else QPalette
    group = QPalette.ColorGroup if hasattr(QPalette, "ColorGroup") else QPalette

    palette = QPalette()
    palette.setColor(role.Window, QColor(53, 53, 53))
    palette.setColor(role.WindowText, QColor(225, 225, 225))
    palette.setColor(role.Base, QColor(35, 35, 35))
    palette.setColor(role.AlternateBase, QColor(53, 53, 53))
    palette.setColor(role.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(role.ToolTipText, QColor(225, 225, 225))
    palette.setColor(role.Text, QColor(225, 225, 225))
    palette.setColor(role.Button, QColor(53, 53, 53))
    palette.setColor(role.ButtonText, QColor(225, 225, 225))
    palette.setColor(role.BrightText, QColor(255, 80, 80))
    palette.setColor(role.Highlight, QColor(42, 130, 218))
    palette.setColor(role.HighlightedText, QColor(20, 20, 20))
    palette.setColor(group.Disabled, role.Text, QColor(120, 120, 120))
    palette.setColor(group.Disabled, role.ButtonText, QColor(120, 120, 120))
    palette.setColor(group.Disabled, role.WindowText, QColor(120, 120, 120))
    app.setPalette(palette)
    app.setStyleSheet(
        "QToolTip { color: #e1e1e1; background-color: #2b2b2b; border: 1px solid #4a4a4a; }"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode Launcher (Qt) for running tape-decode commands"
    )
    parser.parse_args(argv)

    app = QApplication(sys.argv)
    _apply_fusion_dark_mode(app)
    window = DecodeLauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
