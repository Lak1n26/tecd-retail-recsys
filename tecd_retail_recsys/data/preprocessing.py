"""Data preprocessing for T-ECD retail dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for T-ECD retail dataset.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize preprocessor with config.
        """
        self.cfg = cfg
        self.raw_data_dir = Path(cfg.paths.raw_data_dir)
        self.processed_data_dir = Path(cfg.paths.processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.user_to_idx: dict[str, int] = {}
        self.idx_to_user: dict[int, str] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_item: dict[int, str] = {}

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load all retail events from parquet files.
        """
        events_dir = self.raw_data_dir / self.cfg.data.domain / "events"

        all_events = []
        day_begin = self.cfg.data.day_begin
        day_end = self.cfg.data.day_end

        logger.info(f"Loading retail events from day {day_begin} to {day_end}")

        for day in tqdm(range(day_begin, day_end + 1), desc="Loading data"):
            file_path = events_dir / f"{day:05d}.pq"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df["day"] = day
                all_events.append(df)

        data = pd.concat(all_events, ignore_index=True)
        logger.info(f"Loaded {len(data):,} total events")

        return data

    def filter_events(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter events by action type.
        """
        action_type = self.cfg.data.action_type
        filtered = data[data["action_type"] == action_type].copy()
        logger.info(f"Filtered to {len(filtered):,} events with action_type='{action_type}'")
        return filtered

    def filter_by_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter users and items by minimum interaction count.
        """
        min_user = self.cfg.data.min_user_interactions
        min_item = self.cfg.data.min_item_interactions
        prev_len = 0
        current_len = len(data)

        while prev_len != current_len:
            prev_len = current_len
            user_counts = data["user_id"].value_counts()
            valid_users = user_counts[user_counts >= min_user].index
            data = data[data["user_id"].isin(valid_users)]
            item_counts = data["item_id"].value_counts()
            valid_items = item_counts[item_counts >= min_item].index
            data = data[data["item_id"].isin(valid_items)]
            current_len = len(data)

        logger.info(
            f"After filtering (min_user={min_user}, min_item={min_item}): "
            f"{len(data):,} events, {data['user_id'].nunique():,} users, "
            f"{data['item_id'].nunique():,} items"
        )

        return data

    def create_mappings(self, data: pd.DataFrame) -> None:
        """
        Create user and item ID mappings.
        """
        unique_users = sorted(data["user_id"].unique())
        unique_items = sorted(data["item_id"].unique())
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        logger.info(
            f"Created mappings: {len(self.user_to_idx)} users, {len(self.item_to_idx)} items"
        )

    def apply_mappings(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ID mappings to data.
        """
        data = data.copy()
        data["user_idx"] = data["user_id"].map(self.user_to_idx)
        data["item_idx"] = data["item_id"].map(self.item_to_idx)
        return data

    def temporal_split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by time (Global Temporal Split).
        """
        val_days = self.cfg.data.val_days
        test_days = self.cfg.data.test_days
        max_day = data["day"].max()
        test_start = max_day - test_days + 1
        val_start = test_start - val_days
        train_data = data[data["day"] < val_start]
        val_data = data[(data["day"] >= val_start) & (data["day"] < test_start)]
        test_data = data[data["day"] >= test_start]

        logger.info(
            f"Temporal split - Train: days < {val_start} ({len(train_data):,} events), "
            f"Val: days {val_start}-{test_start - 1} ({len(val_data):,} events), "
            f"Test: days >= {test_start} ({len(test_data):,} events)"
        )

        return train_data, val_data, test_data

    def create_user_sequences(self, data: pd.DataFrame) -> dict[int, list[int]]:
        """
        Create user interaction sequences sorted by time.
        """
        data_sorted = data.sort_values(["day", "timestamp"])
        sequences = {}
        for user_idx, group in data_sorted.groupby("user_idx"):
            items = group["item_idx"].tolist()
            max_len = self.cfg.data.max_sequence_length
            if len(items) > max_len:
                items = items[-max_len:]
            sequences[user_idx] = items

        return sequences

    def compute_item_popularity(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute item popularity scores for negative sampling.
        """
        num_items = len(self.item_to_idx)
        popularity = np.zeros(num_items)
        item_counts = data["item_idx"].value_counts()
        for item_idx, count in item_counts.items():
            popularity[item_idx] = count
        popularity = np.power(popularity, 0.75)
        popularity = popularity / popularity.sum()
        return popularity

    def process(self) -> dict:
        """
        Run full preprocessing pipeline.
        """
        logger.info("Starting data preprocessing...")

        data = self.load_raw_data()
        data = self.filter_events(data)
        data = self.filter_by_interactions(data)

        self.create_mappings(data)
        data = self.apply_mappings(data)

        train_data, val_data, test_data = self.temporal_split(data)

        train_sequences = self.create_user_sequences(train_data)
        val_sequences = self.create_user_sequences(val_data)
        test_sequences = self.create_user_sequences(test_data)

        item_popularity = self.compute_item_popularity(train_data)
        val_ground_truth = self._create_ground_truth(val_data)
        test_ground_truth = self._create_ground_truth(test_data)
        train_user_items = self._get_user_items(train_data)

        processed_data = {
            "train_sequences": train_sequences,
            "val_sequences": val_sequences,
            "test_sequences": test_sequences,
            "train_user_items": train_user_items,
            "val_ground_truth": val_ground_truth,
            "test_ground_truth": test_ground_truth,
            "item_popularity": item_popularity,
            "num_users": len(self.user_to_idx),
            "num_items": len(self.item_to_idx),
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
        }

        self._save_processed_data(processed_data)

        logger.info("Preprocessing complete!")
        return processed_data

    def _create_ground_truth(self, data: pd.DataFrame) -> dict[int, set[int]]:
        """
        Create ground truth dict mapping user_idx to set of item_idx.
        """
        ground_truth = {}
        for user_idx, group in data.groupby("user_idx"):
            ground_truth[user_idx] = set(group["item_idx"].unique())
        return ground_truth

    def _get_user_items(self, data: pd.DataFrame) -> dict[int, set[int]]:
        """
        Get set of items for each user.
        """
        user_items = {}
        for user_idx, group in data.groupby("user_idx"):
            user_items[user_idx] = set(group["item_idx"].unique())
        return user_items

    def _save_processed_data(self, data: dict) -> None:
        """
        Save processed data to disk.
        """
        import pickle

        output_path = self.processed_data_dir / "processed_data.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved processed data to {output_path}")

    @classmethod
    def load_processed_data(cls, processed_data_dir: Path) -> dict:
        """
        Load previously processed data.
        """
        import pickle

        input_path = Path(processed_data_dir) / "processed_data.pkl"
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded processed data from {input_path}")
        return data
