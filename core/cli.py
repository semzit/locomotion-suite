from __future__ import annotations

import argparse

from .manifest import load_manifest
from .runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(prog="locomotion")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate an experiment manifest")
    validate.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        manifest = load_manifest(args.manifest)
        ExperimentRunner().validate(manifest)
        print(f"Valid manifest: {manifest.experiment.name}")


if __name__ == "__main__":
    main()
