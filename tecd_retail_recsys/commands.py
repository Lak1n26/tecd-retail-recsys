"""CLI commands for T-ECD Retail Recommendation System."""

import logging
import sys
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_config(config_path: str | None = None, overrides: list[str] | None = None) -> DictConfig:
    """
    Load Hydra configuration.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = str((project_root / "configs").resolve())
    else:
        config_path = str(Path(config_path).resolve())

    if overrides is None:
        overrides = []

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    return cfg


def _parse_overrides(override_str: str | None) -> list[str]:
    """
    Parse override string to list of overrides.
    """
    if not override_str:
        return []
    return [o.strip() for o in override_str.split(",") if o.strip()]


class Commands:
    """
    CLI commands for the recommendation system.
    """

    def preprocess(
        self,
        config_path: str | None = None,
        overrides: str | None = None,
    ) -> None:
        """
        Run data preprocessing.
        """
        override_list = _parse_overrides(overrides)
        cfg = get_config(config_path, override_list)

        from tecd_retail_recsys.data.preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(cfg)
        preprocessor.process()
        logger.info("Preprocessing complete!")

    def train(
        self,
        config_path: str | None = None,
        overrides: str | None = None,
    ) -> None:
        """
        Train the recommendation model.
        """
        override_list = _parse_overrides(overrides)
        cfg = get_config(config_path, override_list)

        from tecd_retail_recsys.train import train

        results = train(cfg)
        logger.info(f"Training results: {results}")

    def infer(
        self,
        checkpoint: str,
        output: str | None = None,
        config_path: str | None = None,
        overrides: str | None = None,
    ) -> None:
        """
        Generate recommendations using trained model.
        """
        override_list = _parse_overrides(overrides)
        cfg = get_config(config_path, override_list)

        from tecd_retail_recsys.infer import infer

        results = infer(cfg, checkpoint, output)
        logger.info(f"Generated recommendations for {results['num_users']} users")

    def evaluate(
        self,
        checkpoint: str,
        config_path: str | None = None,
        overrides: str | None = None,
    ) -> None:
        """
        Evaluate model on test set.
        """
        override_list = _parse_overrides(overrides)
        cfg = get_config(config_path, override_list)

        import pytorch_lightning as pl

        from tecd_retail_recsys.data import RecommendationDataModule
        from tecd_retail_recsys.models import MLPRanker

        data_module = RecommendationDataModule(cfg)
        data_module.prepare_data()
        data_module.setup("test")

        model = MLPRanker.load_from_checkpoint(
            checkpoint,
            num_users=data_module.num_users,
            num_items=data_module.num_items,
            cfg=cfg,
        )

        trainer = pl.Trainer(
            accelerator=cfg.train.trainer.accelerator,
            devices=cfg.train.trainer.devices,
        )
        results = trainer.test(model, data_module)
        logger.info(f"Test results: {results}")

    def download_data(self) -> None:
        """
        Download data from DVC remote storage.
        """
        import subprocess

        logger.info("Pulling data from DVC remote...")
        result = subprocess.run(
            ["dvc", "pull"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("Data downloaded successfully!")
        else:
            logger.error(f"Failed to download data: {result.stderr}")
            raise RuntimeError("DVC pull failed")


def main() -> None:
    """
    Main entry point for CLI.
    """
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
