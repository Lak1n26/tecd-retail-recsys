"""MLP-based ranking model for recommendations."""

import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class MLPRankerModule(nn.Module):
    """
    MLP module for computing user-item interaction scores.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_embedding_dim: int = 64,
        item_embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """
        Initialize MLP Ranker.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim)

        input_dim = user_embedding_dim + item_embedding_dim
        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize embedding weights.
        """
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for user-item pairs.
        """
        original_shape = item_idx.shape
        if len(original_shape) > 1:
            batch_size, num_items = original_shape
            user_idx = user_idx.unsqueeze(1).expand(-1, num_items).reshape(-1)
            item_idx = item_idx.reshape(-1)

        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        concat = torch.cat([user_emb, item_emb], dim=-1)
        scores = self.mlp(concat).squeeze(-1)  # [N]
        if len(original_shape) > 1:
            scores = scores.reshape(original_shape)
        return scores

    def get_all_item_scores(self, user_idx: torch.Tensor, chunk_size: int = 10000) -> torch.Tensor:
        """
        Get scores for all items for given users.
        """
        batch_size = user_idx.shape[0]
        device = user_idx.device
        all_scores = []
        for start_idx in range(0, self.num_items, chunk_size):
            end_idx = min(start_idx + chunk_size, self.num_items)
            chunk_items = torch.arange(start_idx, end_idx, device=device)
            chunk_items = chunk_items.unsqueeze(0).expand(batch_size, -1)
            chunk_scores = self.forward(user_idx, chunk_items)
            all_scores.append(chunk_scores)

        return torch.cat(all_scores, dim=1)


class MLPRanker(pl.LightningModule):
    """
    PyTorch Lightning module for MLP Ranker.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        cfg: DictConfig,
    ):
        """
        Initialize Lightning module.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])

        self.cfg = cfg
        self.num_users = num_users
        self.num_items = num_items

        self.model = MLPRankerModule(
            num_users=num_users,
            num_items=num_items,
            user_embedding_dim=cfg.model.user_embedding_dim,
            item_embedding_dim=cfg.model.item_embedding_dim,
            hidden_dims=list(cfg.model.hidden_dims),
            dropout=cfg.model.dropout,
            use_batch_norm=cfg.model.use_batch_norm,
        )

        self.eval_k = list(cfg.train.eval_k)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.model(user_idx, item_idx)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step with BPR loss.
        """
        user_idx = batch["user_idx"]
        pos_item_idx = batch["pos_item_idx"]
        neg_item_idx = batch["neg_item_idx"]
        pos_scores = self.model(user_idx, pos_item_idx)
        neg_scores = self.model(user_idx, neg_item_idx)
        pos_scores = pos_scores.unsqueeze(1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        """
        Validation step - compute metrics.
        """
        return self._eval_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        """
        Test step - compute metrics.
        """
        return self._eval_step(batch, "test")

    def _eval_step(self, batch: dict[str, torch.Tensor], prefix: str) -> dict[str, Any]:
        """
        Evaluation step for validation or test.
        """
        user_idx = batch["user_idx"]
        ground_truth = batch["ground_truth"]
        num_gt = batch["num_ground_truth"]
        batch_size = user_idx.shape[0]
        all_scores = self.model.get_all_item_scores(user_idx)

        metrics = {}
        for k in self.eval_k:
            hit_rates = []
            precisions = []
            recalls = []
            ndcgs = []

            _, top_k_items = torch.topk(all_scores, k, dim=1)
            for i in range(batch_size):
                gt_set = set(ground_truth[i, : num_gt[i]].cpu().numpy())
                pred_set = set(top_k_items[i].cpu().numpy())

                # Hit Rate
                hits = len(gt_set & pred_set)
                hit_rates.append(1.0 if hits > 0 else 0.0)

                # Precision@k
                precisions.append(hits / k)

                # Recall@k
                recalls.append(hits / len(gt_set) if len(gt_set) > 0 else 0.0)

                # NDCG@k
                ndcg = self._compute_ndcg(top_k_items[i].cpu().numpy(), gt_set, k)
                ndcgs.append(ndcg)

            metrics[f"{prefix}_hit_rate_{k}"] = np.mean(hit_rates)
            metrics[f"{prefix}_precision_{k}"] = np.mean(precisions)
            metrics[f"{prefix}_recall_{k}"] = np.mean(recalls)
            metrics[f"{prefix}_ndcg_{k}"] = np.mean(ndcgs)

        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, prog_bar=("hit_rate_100" in name))

        return metrics

    def _compute_ndcg(self, ranked_items: np.ndarray, ground_truth: set, k: int) -> float:
        """
        Compute NDCG@k.
        """
        dcg = 0.0
        for i, item in enumerate(ranked_items[:k]):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def configure_optimizers(self) -> dict:
        """
        Configure optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )

        scheduler_cfg = self.cfg.train.scheduler
        t_max = max(1, self.cfg.train.epochs - scheduler_cfg.warmup_epochs)
        if scheduler_cfg.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=scheduler_cfg.min_lr,
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
