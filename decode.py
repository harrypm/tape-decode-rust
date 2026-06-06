#!/usr/bin/env python3
from __future__ import annotations

import sys
from multiprocessing import freeze_support

from decode_runtime import run_tape_decode


LAUNCHER_ALIASES = {"decode-launcher", "decode_launcher", "launcher", "gui"}


def main(argv: list[str]) -> int:
    args = list(argv)
    if not args:
        from decode_launcher import main as launcher_main

        return launcher_main([])

    first = args[0].lower()
    if first in LAUNCHER_ALIASES:
        from decode_launcher import main as launcher_main

        return launcher_main(args[1:])

    try:
        return run_tape_decode(args)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main(sys.argv[1:]))
