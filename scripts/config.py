"""
Global configuration file reader.
Simplified version - most settings are now provided via CLI arguments or Gradio UI.
"""

import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from style_bert_vits2.logging import logger


class PathConfig:
    """Path configuration from configs/paths.yml"""

    def __init__(self, dataset_root: str, assets_root: str):
        self.dataset_root = Path(dataset_root)
        self.assets_root = Path(assets_root)


# Check CUDA availability
cuda_available = torch.cuda.is_available()


class TrainConfig:
    """Training configuration"""

    def __init__(self, spec_cache: bool = True, keep_ckpts: int = 3):
        self.spec_cache = spec_cache
        self.keep_ckpts = keep_ckpts

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


class ServerConfig:
    """API server configuration (for server_fastapi.py)"""

    def __init__(
        self,
        port: int = 5000,
        device: str = "cuda",
        limit: int = 100,
        language: str = "JP",
        origins: list[str] = None,
    ):
        self.port = port
        if not cuda_available:
            device = "cpu"
        self.device = device
        self.language = language
        self.limit = limit
        self.origins = origins or ["*"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


class Config:
    """Main configuration class"""

    def __init__(self, config_path: str, path_config: PathConfig):
        if not Path(config_path).exists():
            shutil.copy(src="default_config.yml", dst=config_path)
            logger.info(
                f"A configuration file {config_path} has been generated based on default_config.yml."
            )

        with open(config_path, encoding="utf-8") as file:
            yaml_config: dict[str, Any] = yaml.safe_load(file)

            # Model name and path
            model_name: str = yaml_config.get("model_name", "")
            self.model_name = model_name

            if "dataset_path" in yaml_config and yaml_config["dataset_path"]:
                dataset_path = Path(yaml_config["dataset_path"])
            else:
                dataset_path = (
                    path_config.dataset_root / model_name
                    if model_name
                    else path_config.dataset_root
                )
            self.dataset_path = dataset_path
            self.dataset_root = path_config.dataset_root
            self.assets_root = path_config.assets_root
            self.out_dir = (
                self.assets_root / model_name if model_name else self.assets_root
            )

            # Global defaults
            device = yaml_config.get("device", "cuda")
            if not cuda_available:
                device = "cpu"
            self.device = device
            self.num_processes = yaml_config.get("num_processes", 4)

            # Training config
            train_data = yaml_config.get("train", {})
            self.train_config = TrainConfig.from_dict(train_data)

            # Server config
            server_data = yaml_config.get("server", {})
            self.server_config = ServerConfig.from_dict(server_data)


def get_path_config() -> PathConfig:
    """Load path configuration from configs/paths.yml"""
    path_config_path = Path("configs/paths.yml")
    if not path_config_path.exists():
        shutil.copy(src="configs/default_paths.yml", dst=path_config_path)
        logger.info(
            f"A configuration file {path_config_path} has been generated based on default_paths.yml."
        )
    with open(path_config_path, encoding="utf-8") as file:
        path_config_dict: dict[str, str] = yaml.safe_load(file)
    return PathConfig(**path_config_dict)


def get_config() -> Config:
    """Load main configuration"""
    path_config = get_path_config()
    try:
        config = Config("config.yml", path_config)
    except (TypeError, KeyError) as e:
        logger.warning(f"Config error: {e}. Regenerating from default_config.yml.")
        shutil.copy(src="default_config.yml", dst="config.yml")
        config = Config("config.yml", path_config)
    return config
