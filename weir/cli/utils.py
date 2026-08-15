from __future__ import annotations

import logging
import os
from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig

from weir.core.utils import CONFIG_DIR

CONFIG_RELATIVE = str(os.path.relpath(CONFIG_DIR, Path(__file__).parent))


def setup_logging() -> None:
    """Configure INFO-level structured logging shared by all entry points."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def compose_config(agent: str, task: str) -> DictConfig:
    """Compose the train config with the requested agent and task groups."""
    with initialize(version_base=None, config_path=CONFIG_RELATIVE):
        return compose(config_name="train", overrides=[f"agent={agent}", f"task={task}"])
