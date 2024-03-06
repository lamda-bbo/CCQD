import hydra
from typing import TYPE_CHECKING

from .main import main
from .config import register_configs

if TYPE_CHECKING:
    from .config import Config


register_configs()


@hydra.main(config_path='../config', config_name='config', version_base=None)
def _module_main(cfg: 'Config'):
    main(cfg)


if __name__ == '__main__':
    _module_main()
