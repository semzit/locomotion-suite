from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from weir.core.contracts import Action, Observation, Shape, SimStep
from weir.core.tasks import TASKS, Task


class MuJoCoSim:
    """MuJoCo-backed SimBackend for single-environment rollouts."""

    def __init__(self) -> None:
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._task: Task | None = None
        self._time_limit = float("inf")
        self._renderer: mujoco.Renderer | None = None

    def load(self, agent_config: dict[str, Any], sim_config: dict[str, Any]) -> None:
        model_path = str(agent_config["model"])
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        task_config = sim_config.get("task", {})
        task_name = str(task_config.get("name", "survive"))
        task_params = dict(task_config.get("params", {}))
        try:
            task_type = TASKS[task_name]
        except KeyError as error:
            raise ValueError(f"Unknown task: {task_name!r}") from error
        self._task = task_type(**task_params)
        self._time_limit = float(sim_config.get("time_limit", float("inf")))

    def reset(self, seed: int | None = None) -> Observation:
        model = self._require_model()
        data = self._require_data()
        mujoco.mj_resetData(model, data)
        if seed is not None and model.nq > 7:
            rng = np.random.default_rng(seed)
            data.qpos[7:] += rng.normal(0.0, 0.05, size=model.nq - 7)
            mujoco.mj_forward(model, data)
        return self._observe(data)

    def step(self, actions: Action) -> SimStep:
        model = self._require_model()
        data = self._require_data()
        task = self._require_task()
        action = np.asarray(actions, dtype=float)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        observation = self._observe(data)
        return SimStep(
            observation=observation,
            reward=float(task.reward(observation, action.astype(np.float32))),
            terminated=bool(task.terminated(observation)),
            truncated=bool(data.time >= self._time_limit),
        )

    def observation_shape(self) -> Shape:
        model = self._require_model()
        return Shape(dims=(model.nq + model.nv,), dtype="float32")

    def action_shape(self) -> Shape:
        model = self._require_model()
        ctrlrange = model.actuator_ctrlrange
        return Shape(
            dims=(model.nu,),
            dtype="float32",
            low=ctrlrange[:, 0].astype(np.float32),
            high=ctrlrange[:, 1].astype(np.float32),
        )

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render the current state with MuJoCo's offscreen renderer (RGB uint8)."""
        model = self._require_model()
        data = self._require_data()
        if (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            try:
                self._renderer = mujoco.Renderer(model, height=height, width=width)
            except Exception as error:
                self._renderer = None
                raise RuntimeError(
                    "Offscreen rendering is unavailable in this environment: could not "
                    f"create a mujoco.Renderer ({type(error).__name__}: {error})"
                ) from error
        mujoco.mj_forward(model, data)
        self._renderer.update_scene(data)
        return self._renderer.render()[..., :3].copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._data = None
        self._model = None

    def domain_params(self) -> dict[str, Any]:
        model = self._require_model()
        return {
            "body_mass": model.body_mass.copy(),
            "geom_friction": model.geom_friction.copy(),
            "dof_damping": model.dof_damping.copy(),
        }

    def apply_domain_params(self, params: dict[str, Any]) -> None:
        model = self._require_model()
        data = self._require_data()
        if "body_mass" in params:
            model.body_mass[:] = params["body_mass"]
        if "geom_friction" in params:
            model.geom_friction[:] = params["geom_friction"]
        if "dof_damping" in params:
            model.dof_damping[:] = params["dof_damping"]
        mujoco.mj_forward(model, data)

    def apply_perturbation(self, force: Action) -> None:
        data = self._require_data()
        data.xfrc_applied[0, :3] = np.asarray(force, dtype=float)[:3]
        data.xfrc_applied[0, 3:] = 0.0

    def _observe(self, data: mujoco.MjData) -> Observation:
        return np.concatenate([data.qpos, data.qvel]).astype(np.float32)

    def _require_model(self) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("MuJoCoSim.load() must be called before use")
        return self._model

    def _require_data(self) -> mujoco.MjData:
        if self._data is None:
            raise RuntimeError("MuJoCoSim.load() must be called before use")
        return self._data

    def _require_task(self) -> Task:
        if self._task is None:
            raise RuntimeError("MuJoCoSim.load() must be called before use")
        return self._task
