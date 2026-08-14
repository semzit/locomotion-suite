from __future__ import annotations

import argparse
from typing import Any

from pydantic import TypeAdapter

PLUGIN_ENTRY_POINT_GROUP = "locomotion_suite.plugins"

CONFIG_MAPPING: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the locomotion CLI."""
    parser = argparse.ArgumentParser(prog="locomotion")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate an experiment manifest")
    validate.add_argument("--manifest", required=True)

    commands.add_parser("plugins", help="List installed plugin entry points")

    run = commands.add_parser(
        "run",
        help="Execute a minimal experiment and emit an artifact bundle",
    )
    run.add_argument("--manifest", required=True)
    return parser
