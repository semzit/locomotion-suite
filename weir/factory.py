from __future__ import annotations

from weir.algo.ppo import PPOAlgorithm
from weir.contracts import AlgorithmPlugin, SimBackend
from weir.envs.sim.mujoco import MuJoCoSim

SIMS: dict[str, type[SimBackend]] = {
    "mujoco": MuJoCoSim,
}

ALGORITHMS: dict[str, type[AlgorithmPlugin]] = {
    "ppo": PPOAlgorithm,
}


def create_sim(name: str) -> SimBackend:
    try:
        return SIMS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown sim backend: {name!r}") from error


def create_algorithm(name: str) -> AlgorithmPlugin:
    try:
        return ALGORITHMS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown algorithm: {name!r}") from error
