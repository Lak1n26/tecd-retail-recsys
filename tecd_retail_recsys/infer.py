"""Inference script for generating recommendations."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from tecd_retail_recsys.data.preprocessing import DataPreprocessor
from tecd_retail_recsys.models import MLPRanker

logger = logging.getLogger(__name__)


class RecommendationInference:
    """
    Inference class for generating recommendations.
    """

    def __init__(
        self,
        checkpoint_path: str,
        cfg: DictConfig,
        device: str = "auto",
    ):
        """
        Initialize inference.
        """
        self.cfg = cfg
        self.checkpoint_path = Path(checkpoint_path)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.processed_data = DataPreprocessor.load_processed_data(cfg.paths.processed_data_dir)
        self.model = self._load_model()

    def _load_model(self) -> MLPRanker:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading model from {self.checkpoint_path}")

        model = MLPRanker.load_from_checkpoint(
            self.checkpoint_path,
            num_users=self.processed_data["num_users"],
            num_items=self.processed_data["num_items"],
            cfg=self.cfg,
        )
        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def recommend(
        self,
        user_ids: list[int | str],
        top_k: int = 100,
        exclude_seen: bool = True,
    ) -> dict[int | str, list[str]]:
        """
        Generate recommendations for users.
        """
        recommendations = {}
        user_to_idx = self.processed_data["user_to_idx"]
        idx_to_item = self.processed_data["idx_to_item"]
        train_user_items = self.processed_data["train_user_items"]

        for user_id in tqdm(user_ids, desc="Generating recommendations"):
            if isinstance(user_id, str) or user_id not in range(self.processed_data["num_users"]):
                if user_id not in user_to_idx:
                    logger.warning(f"User {user_id} not found in training data")
                    recommendations[user_id] = []
                    continue
                user_idx = user_to_idx[user_id]
            else:
                user_idx = user_id

            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            scores = self.model.model.get_all_item_scores(user_tensor)
            scores = scores.squeeze(0).cpu().numpy()

            if exclude_seen and user_idx in train_user_items:
                seen_items = train_user_items[user_idx]
                for item_idx in seen_items:
                    scores[item_idx] = float("-inf")

            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            top_k_items = [idx_to_item[idx] for idx in top_k_indices]
            recommendations[user_id] = top_k_items
        return recommendations

    @torch.no_grad()
    def recommend_batch(
        self,
        user_indices: list[int],
        top_k: int = 100,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Generate recommendations for users in batches.
        """
        all_recommendations = []

        for i in tqdm(range(0, len(user_indices), batch_size), desc="Batch inference"):
            batch_users = user_indices[i : i + batch_size]
            user_tensor = torch.tensor(batch_users, dtype=torch.long, device=self.device)
            scores = self.model.model.get_all_item_scores(user_tensor)
            scores = scores.cpu().numpy()
            batch_top_k = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]
            all_recommendations.append(batch_top_k)

        return np.vstack(all_recommendations)

    def save_recommendations(
        self,
        recommendations: dict[int | str, list[str]],
        output_path: str,
    ) -> None:
        """
        Save recommendations to CSV file.
        """
        rows = []
        for user_id, items in recommendations.items():
            for rank, item_id in enumerate(items, 1):
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rank": rank,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved recommendations to {output_path}")


def infer(cfg: DictConfig, checkpoint_path: str, output_path: str | None = None) -> dict:
    """
    Run inference pipeline.
    """
    logger.info("Starting inference pipeline...")
    inference = RecommendationInference(checkpoint_path, cfg)
    processed_data = inference.processed_data
    val_users = list(processed_data["val_ground_truth"].keys())
    logger.info(f"Generating recommendations for {len(val_users)} users")
    recommendations = inference.recommend(val_users, top_k=100)

    if output_path:
        inference.save_recommendations(recommendations, output_path)
    logger.info("Inference complete!")
    return {"num_users": len(recommendations), "recommendations": recommendations}
