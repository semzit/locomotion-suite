from __future__ import annotations

import argparse


PLUGIN_ENTRY_POINT_GROUP = "locomotion_suite.plugins"

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the locomotion CLI."""
    parser = argparse.ArgumentParser(prog="locomotion")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate an experiment manifest")
    validate.add_argument("--manifest", required=True)

    commands.add_parser("plugins", help="List installed plugin entry points")
    return parser
