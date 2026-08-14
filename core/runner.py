from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from core.contracts.algorithm import AlgorithmPlugin
from core.contracts.artifacts import ArtifactBundle
from core.contracts.compute import ComputeBackend
from core.contracts.sim import SimBackend
from core.contracts.transitions import TransitionBatch
from core.registry import PluginRegistry

from .manifest import ExperimentManifest


class ExperimentRunner:
    """Coordinates a resolved experiment; execution arrives with the reference stack."""

    def validate(self, manifest: ExperimentManifest) -> None:
        if manifest.experiment.mode not in {"train", "eval", "export", "search"}:
            raise ValueError(f"Unsupported run mode: {manifest.experiment.mode}")

    def run(
        self,
        manifest: ExperimentManifest,
        output_dir: str | Path | None = None,
    ) -> ArtifactBundle:
        self.validate(manifest)

        extras = manifest.model_extra or {}
        sim_config = dict(extras.get("sim", {}))
        algorithm_config = dict(extras.get("algorithm", {}))
        compute_config = dict(extras.get("compute", {}))

        sim_plugin_name = str(sim_config.get("plugin", "mujoco"))
        algorithm_plugin_name = str(algorithm_config.get("plugin", "ppo"))
        compute_plugin_name = str(compute_config.get("plugin", "local"))

        registry = PluginRegistry()
        sim_backend = cast(SimBackend, registry.create(sim_plugin_name))
        algorithm = cast(AlgorithmPlugin, registry.create(algorithm_plugin_name))
        compute = cast(ComputeBackend, registry.create(compute_plugin_name))

        sim_backend.load({"name": manifest.experiment.name}, sim_config)
        observation_spec = sim_backend.observation_spec()
        action_spec = sim_backend.action_spec()
        algorithm.configure(observation_spec, action_spec, algorithm_config)

        initial_obs = sim_backend.reset(batch_size=1)
        actions = algorithm.sample_actions(initial_obs, deterministic=True)
        transition = TransitionBatch(
            observations=initial_obs,
            actions=actions,
            rewards=[0.0],
            next_observations=initial_obs,
            terminated=[False],
            truncated=[False],
            info={"mode": manifest.experiment.mode},
            metadata={"plugin": algorithm_plugin_name},
        )
        metrics = algorithm.train_step(transition)

        if output_dir is not None:
            artifact_root = Path(output_dir).resolve()
        else:
            artifact_root = Path.cwd() / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_root / "manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

        policy_path = artifact_root / "policy.bin"
        algorithm.save(policy_path)
        export_bytes = algorithm.export("onnx")
        export_path = artifact_root / "policy.onnx"
        export_path.write_bytes(export_bytes)

        job_handle = compute.launch({"mode": manifest.experiment.mode, "policy": str(policy_path)})
        bundle = ArtifactBundle(
            root=artifact_root,
            manifest_snapshot=manifest_path,
            files={
                "policy": policy_path,
                "export": export_path,
                "metrics": artifact_root / "metrics.json",
            },
            metadata={
                "mode": manifest.experiment.mode,
                "sim_plugin": sim_plugin_name,
                "algorithm_plugin": algorithm_plugin_name,
                "compute_plugin": compute_plugin_name,
                "job_handle": str(job_handle),
                "loss": str(metrics.get("loss", 0.0)),
                "reward": str(metrics.get("reward", 0.0)),
            },
        )

        metrics_path = bundle.files["metrics"]
        metrics_path.write_text(yaml.safe_dump(metrics, sort_keys=False), encoding="utf-8")
        return bundle
