from __future__ import annotations

from .manifest import load_manifest
from .observability import logger, tracer
from .registry import PluginRegistry
from .runner import ExperimentRunner
from .utils import build_parser

# Optional Hydra integration (prototype) — imports are deferred inside the entrypoint


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        with tracer.start_as_current_span("manifest.validate"):
            manifest = load_manifest(args.manifest)
            ExperimentRunner().validate(manifest)
        logger.info("manifest.validated", experiment=manifest.experiment.name)
        print(f"Valid manifest: {manifest.experiment.name}")
    elif args.command == "plugins":
        for name in PluginRegistry().discover():
            print(name)
    elif args.command == "run":
        with tracer.start_as_current_span("experiment.run"):
            manifest = load_manifest(args.manifest)
            bundle = ExperimentRunner().run(manifest)
        logger.info(
            "experiment.completed",
            experiment=manifest.experiment.name,
            output_dir=str(bundle.root),
            mode=manifest.experiment.mode,
        )
        print(f"Ran {manifest.experiment.name} -> {bundle.root}")


if __name__ == "__main__":
    main()


def hydra_main() -> None:
    """Hydra-backed entrypoint prototype.

    This wrapper defers importing `hydra` until runtime so the package can be
    imported without Hydra installed. It defines and invokes a small Hydra
    application that expects a `manifest` override (for example:
    `locomotion-hydra --manifest=path/to/manifest.yaml`).
    """
    try:
        import hydra
        from hydra.utils import to_absolute_path
        from omegaconf import DictConfig
    except Exception as exc:  # pragma: no cover - runtime optional dependency
        raise RuntimeError(
            "Hydra is not available. Install with `pip install hydra-core` to use this entrypoint."
        ) from exc

    @hydra.main(version_base=None, config_path=None)
    def _inner(cfg: DictConfig) -> None:
        manifest = cfg.get("manifest")
        if not manifest:
            print("Provide a manifest with --manifest=path/to/manifest.yaml")
            return

        manifest_path = to_absolute_path(str(manifest))

        with tracer.start_as_current_span("experiment.run.hydra"):
            manifest_obj = load_manifest(manifest_path)
            bundle = ExperimentRunner().run(manifest_obj)

        logger.info(
            "experiment.completed",
            experiment=manifest_obj.experiment.name,
            output_dir=str(bundle.root),
            mode=manifest_obj.experiment.mode,
        )
        print(f"Ran {manifest_obj.experiment.name} -> {bundle.root}")

    _inner()
