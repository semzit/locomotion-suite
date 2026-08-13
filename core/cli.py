from __future__ import annotations

from .manifest import load_manifest
from .observability import logger, tracer
from .registry import PluginRegistry
from .runner import ExperimentRunner
from .utils import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        with tracer.start_as_current_span("manifest.validate"):
            manifest = load_manifest(args.manifest)
            ExperimentRunner().validate(manifest)
        logger.info("manifest.validated", experiment=manifest.experiment.name)
        print(f"Valid manifest: {manifest.experiment.name}")
    if args.command == "plugins":
        for name in PluginRegistry().discover():
            print(name)


if __name__ == "__main__":
    main()
