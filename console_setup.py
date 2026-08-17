"""
console_setup.py  —  Make console output safe on any terminal.

The scripts print box-drawing characters and emoji. On Windows the default
console codepage is cp1252, which cannot encode them, and printing raises
UnicodeEncodeError — killing a training run over a progress message.

Importing this module forces stdout/stderr to UTF-8 and, failing that, replaces
unencodable characters instead of raising. Imported by config.py, so every
entry point that imports config gets it for free.
"""

import sys


def configure_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # redirected to something without reconfigure(); nothing to do
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Last resort: keep the terminal's encoding but never raise on output.
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


configure_console()
