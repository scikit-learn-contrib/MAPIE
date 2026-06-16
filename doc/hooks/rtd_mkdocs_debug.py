"""Run MkDocs on Read the Docs with unbuffered diagnostic output."""

from __future__ import annotations

import faulthandler
import os
import platform
import sys
import time


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[rtd-mkdocs-debug {timestamp}] {message}", flush=True)


def main() -> None:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)

    output_dir = os.environ.get("READTHEDOCS_OUTPUT", "_readthedocs_output")
    args = [
        "build",
        "--verbose",
        "--clean",
        "--site-dir",
        f"{output_dir}/html",
        "--config-file",
        "mkdocs.yml",
    ]

    _log(f"cwd={os.getcwd()}")
    _log(f"python={sys.version.replace(os.linesep, ' ')}")
    _log(f"platform={platform.platform()}")
    _log(f"READTHEDOCS_OUTPUT={output_dir}")
    _log("importing mkdocs")

    import mkdocs
    import mkdocs.__main__

    _log(f"mkdocs={mkdocs.__version__}")
    _log(f"calling mkdocs {' '.join(args)}")

    try:
        mkdocs.__main__.cli(args=args, prog_name="mkdocs", standalone_mode=True)
    finally:
        _log("mkdocs command returned or raised")


if __name__ == "__main__":
    main()
