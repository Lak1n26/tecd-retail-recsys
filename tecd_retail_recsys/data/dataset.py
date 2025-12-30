"""PyTorch Dataset and DataModule for recommendation system."""

import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from tecd_retail_recsys.data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class RecommendationDataset(Dataset):
    """
    Dataset for training recommendation model.
    """

    def __init__(
        self,
        user_sequences: dict[int, list[int]],
        user_items: dict[int, set[int]],
        num_items: int,
        item_popularity: np.ndarray,
        num_negatives: int = 4,
        is_training: bool = True,
    ):
        """
        Initialize dataset.
        """
        self.user_sequences = user_sequences
        self.user_items = user_items
        self.num_items = num_items
        self.item_popularity = item_popularity
        self.num_negatives = num_negatives
        self.is_training = is_training
        self.samples = []
        for user_idx, items in user_sequences.items():
            for item_idx in items:
                self.samples.append((user_idx, item_idx))

        logger.info(f"Created dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_idx, pos_item_idx = self.samples[idx]
        neg_items = self._sample_negatives(user_idx)

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "pos_item_idx": torch.tensor(pos_item_idx, dtype=torch.long),
            "neg_item_idx": torch.tensor(neg_items, dtype=torch.long),
        }

    def _sample_negatives(self, user_idx: int) -> list[int]:
        """
        Sample negative items for a user.
        """
        user_items = self.user_items.get(user_idx, set())
        neg_items = []
        adjusted_pop = self.item_popularity.copy()
        for item in user_items:
            adjusted_pop[item] = 0
        adjusted_pop = adjusted_pop / adjusted_pop.sum()
        neg_candidates = np.random.choice(
            self.num_items,
            size=self.num_negatives * 2,
            replace=False,
            p=adjusted_pop,
        )

        for item in neg_candidates:
            if item not in user_items:
                neg_items.append(item)
                if len(neg_items) >= self.num_negatives:
                    break

        while len(neg_items) < self.num_negatives:
            item = np.random.randint(0, self.num_items)
            if item not in user_items and item not in neg_items:
                neg_items.append(item)

        return neg_items


class EvalDataset(Dataset):
    """
    Dataset for evaluation - generates all items for ranking.
    """

    def __init__(
        self,
        user_sequences: dict[int, list[int]],
        ground_truth: dict[int, set[int]],
        num_items: int,
    ):
        """
        Initialize evaluation dataset.
        """
        self.num_items = num_items
        self.ground_truth = ground_truth
        self.users = [u for u in user_sequences.keys() if u in ground_truth]
        self.user_sequences = {u: user_sequences[u] for u in self.users}
        logger.info(f"Created eval dataset with {len(self.users)} users")

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_idx = self.users[idx]
        gt_items = list(self.ground_truth[user_idx])

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "ground_truth": torch.tensor(gt_items, dtype=torch.long),
            "num_ground_truth": torch.tensor(len(gt_items), dtype=torch.long),
        }


def eval_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Custom collate function for evaluation that handles variable-length ground truth.
    """
    user_idx = torch.stack([b["user_idx"] for b in batch])
    num_gt = torch.stack([b["num_ground_truth"] for b in batch])
    max_gt_len = max(b["num_ground_truth"].item() for b in batch)
    ground_truth = torch.zeros(len(batch), max_gt_len, dtype=torch.long)
    for i, b in enumerate(batch):
        gt_len = b["num_ground_truth"].item()
        ground_truth[i, :gt_len] = b["ground_truth"]

    return {
        "user_idx": user_idx,
        "ground_truth": ground_truth,
        "num_ground_truth": num_gt,
    }


class RecommendationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for recommendation system.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize DataModule.
        """
        super().__init__()
        self.cfg = cfg
        self.processed_data: dict | None = None

    def prepare_data(self) -> None:
        """
        Prepare data (run preprocessing if needed).
        """
        processed_path = Path(self.cfg.paths.processed_data_dir) / "processed_data.pkl"

        if not processed_path.exists():
            logger.info("Processed data not found, running preprocessing...")
            preprocessor = DataPreprocessor(self.cfg)
            self.processed_data = preprocessor.process()
        else:
            logger.info("Loading existing processed data...")
            self.processed_data = DataPreprocessor.load_processed_data(
                self.cfg.paths.processed_data_dir
            )

    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets for training/validation/testing.
        """
        if self.processed_data is None:
            self.processed_data = DataPreprocessor.load_processed_data(
                self.cfg.paths.processed_data_dir
            )

        if stage == "fit" or stage is None:
            self.train_dataset = RecommendationDataset(
                user_sequences=self.processed_data["train_sequences"],
                user_items=self.processed_data["train_user_items"],
                num_items=self.processed_data["num_items"],
                item_popularity=self.processed_data["item_popularity"],
                num_negatives=self.cfg.data.num_negatives,
                is_training=True,
            )

            self.val_dataset = EvalDataset(
                user_sequences=self.processed_data["train_sequences"],
                ground_truth=self.processed_data["val_ground_truth"],
                num_items=self.processed_data["num_items"],
            )

        if stage == "test" or stage is None:
            self.test_dataset = EvalDataset(
                user_sequences=self.processed_data["train_sequences"],
                ground_truth=self.processed_data["test_ground_truth"],
                num_items=self.processed_data["num_items"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            collate_fn=eval_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            collate_fn=eval_collate_fn,
        )

    @property
    def num_users(self) -> int:
        """Return number of users."""
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call prepare_data() first.")
        return self.processed_data["num_users"]

    @property
    def num_items(self) -> int:
        """Return number of items."""
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call prepare_data() first.")
        return self.processed_data["num_items"]
