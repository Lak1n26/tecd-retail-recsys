"""Training script for recommendation model."""

import logging
import subprocess

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from tecd_retail_recsys.data import RecommendationDataModule
from tecd_retail_recsys.models import MLPRanker

logger = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    """
    Get current git commit ID.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _try_setup_mlflow(cfg: DictConfig):
    """
    Try to setup MLflow logger, return None if not available.
    """
    try:
        import socket

        host = cfg.mlflow.tracking_uri.replace("http://", "").replace("https://", "")
        host, port = host.split(":")
        port = int(port)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            logger.warning(
                f"MLflow server not available at {cfg.mlflow.tracking_uri}. "
                "Logging to CSV instead."
            )
            return None

        from pytorch_lightning.loggers import MLFlowLogger

        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            log_model=True,
        )
        mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        mlflow_logger.log_hyperparams({"git_commit": get_git_commit_id()})
        return mlflow_logger

    except Exception as e:
        logger.warning(f"Failed to setup MLflow: {e}. Logging to CSV instead.")
        return None


def train(cfg: DictConfig) -> dict:
    """
    Run training pipeline.
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Starting training pipeline...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info("Initializing data module...")

    data_module = RecommendationDataModule(cfg)
    data_module.prepare_data()
    data_module.setup("fit")

    logger.info("Initializing model...")
    model = MLPRanker(
        num_users=data_module.num_users,
        num_items=data_module.num_items,
        cfg=cfg,
    )

    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Num users: {data_module.num_users:,}")
    logger.info(f"Num items: {data_module.num_items:,}")

    pl_logger = _try_setup_mlflow(cfg)
    if pl_logger is None:
        from pytorch_lightning.loggers import CSVLogger

        pl_logger = CSVLogger(save_dir=cfg.paths.output_dir, name="logs")

    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints_dir,
            filename="{epoch}-{val_hit_rate_100:.4f}",
            monitor=cfg.train.checkpoint_monitor,
            mode="max",
            save_top_k=cfg.train.save_top_k,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
            mode=cfg.train.early_stopping.mode,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        precision=cfg.train.trainer.precision,
        logger=pl_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.train.trainer.log_every_n_steps,
        val_check_interval=cfg.train.trainer.val_check_interval,
        deterministic=True,
    )

    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Running test evaluation...")
    data_module.setup("test")
    test_results = trainer.test(model, data_module, ckpt_path="best")

    if isinstance(pl_logger, type(None)) is False:
        try:
            import mlflow

            if mlflow.active_run():
                for key, value in test_results[0].items():
                    mlflow.log_metric(f"final_{key}", value)
        except Exception:
            pass

    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"Test results: {test_results}")

    return {
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
        "test_results": test_results,
    }
