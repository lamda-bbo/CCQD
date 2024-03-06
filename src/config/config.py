from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any

from .algo_config import AlgoConfig, register_algo_configs
from .env_config import EnvConfig, register_env_configs
from ..utils import dataclass_field


_defaults = [
    {'algo': MISSING},
    {'env': MISSING},
]


@dataclass
class Config:
    defaults: list[Any] = dataclass_field(_defaults)

    algo: AlgoConfig = MISSING
    env: EnvConfig = MISSING
    seed: int = MISSING
    code: str = 'latest'
    run: str = 'normal'
    wandb: bool = True
    total_plots: int = 50
    representation_diversity_verification: bool = False
    config_filename: str = 'cfg.yaml'
    metrics_filename: str = 'metrics.csv'
    checkpoint_filename: str = 'checkpoint.pkl.lz4'
    compressed_checkpoint_filename: str = 'checkpoint.pkl.xz'
    representation_diversity_filename: str = 'representation_diversity.pkl.xz'
    checkpoint_saving_interval: int = 100
    metrics_uploading_interval: int = 50
    tmpfile_postfix: str = '~'
    type: str = 'main'


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name='base_config', node=Config)
    register_algo_configs()
    register_env_configs()
