from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class EnvConfig:
    name: str = MISSING
    episode_len: int = 250
    total_steps: int = 150_000_000
    log_period: int = 8000


_env_configs: dict[str, type[EnvConfig]] = {}


_TEnvConfig = TypeVar('_TEnvConfig', bound=EnvConfig)


def _register(cls: type[_TEnvConfig]) -> type[_TEnvConfig]:
    assert cls.name not in _env_configs
    _env_configs[cls.name] = cls
    return cls


@dataclass
class QDaxEnvConfig(EnvConfig):
    pass


@_register
@dataclass
class HopperUniEnvConfig(QDaxEnvConfig):
    name: str = 'hopper_uni'
    log_period: int = 1000


@_register
@dataclass
class Walker2DUniEnvConfig(QDaxEnvConfig):
    name: str = 'walker2d_uni'
    log_period: int = 1000


@_register
@dataclass
class AntUniEnvConfig(QDaxEnvConfig):
    name: str = 'ant_uni'
    log_period: int = 1000


@_register
@dataclass
class HalfCheetahUniEnvConfig(QDaxEnvConfig):
    name: str = 'halfcheetah_uni'
    log_period: int = 500


@_register
@dataclass
class HumanoidUniEnvConfig(QDaxEnvConfig):
    name: str = 'humanoid_uni'
    log_period: int = 500


@_register
@dataclass
class HumanoidOmniEnvConfig(QDaxEnvConfig):
    name: str = 'humanoid_omni'
    log_period: int = 500


@_register
@dataclass
class PointMazeEnvConfig(QDaxEnvConfig):
    name: str = 'pointmaze'
    log_period: int = 1000


@_register
@dataclass
class AntMazeEnvConfig(QDaxEnvConfig):
    name: str = 'antmaze'
    log_period: int = 500


@dataclass
class GymEnvConfig(EnvConfig):
    package: str = 'gym'


@dataclass
class GymnasiumEnvConfig(GymEnvConfig):
    package: str = 'gymnasium'


@dataclass
class AtariEnvConfig(GymnasiumEnvConfig):
    episode_len: int = 2000
    log_period: int = 20


@_register
@dataclass
class PongEnvConfig(AtariEnvConfig):
    name: str = 'ALE--Pong-v5'


def register_env_configs():
    cs = ConfigStore.instance()
    for name, cls in _env_configs.items():
        cs.store(
            group='env',
            name=name,
            node=cls,
        )
