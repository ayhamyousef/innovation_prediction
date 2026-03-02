"""
PyTorch Dataset for patent innovation prediction.

Each sample represents a technology (CPC/IPC code pair) with:
  - Text: concatenated titles of early invention patents
          (abstracts not available from ODP API)
  - Metadata: 7 numerical features from the paper + patent-level features
  - Label: cluster assignment (0-3)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

logger = logging.getLogger("innovation_prediction.dataset")


class PatentDataset(Dataset):
    """
    Dataset that provides tokenized text + metadata for each technology.

    For each technology, we aggregate text from early invention patents
    (or the first/representative patent) and combine it with the 7
    numerical features from the paper.
    """

    def __init__(self, tech_df: pd.DataFrame, patent_lookup: Dict,
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 use_metadata: bool = True,
                 label_col: str = "cluster",
                 metadata_cols: Optional[List[str]] = None):
        """
        Args:
            tech_df: DataFrame with tech features and cluster labels
            patent_lookup: {patent_number: patent_dict}
            tokenizer_name: HuggingFace tokenizer name
            max_length: max token length
            use_metadata: whether to include numerical features
            label_col: column name for labels
            metadata_cols: which columns to use as metadata features
        """
        self.df = tech_df.reset_index(drop=True)
        self.patent_lookup = patent_lookup
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.use_metadata = use_metadata
        self.label_col = label_col

        self.metadata_cols = metadata_cols or [
            "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
            "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
        ]

        # Precompute metadata normalization stats
        if use_metadata:
            existing_cols = [c for c in self.metadata_cols if c in self.df.columns]
            self.metadata_cols = existing_cols
            if existing_cols:
                meta_values = self.df[existing_cols].values.astype(np.float32)
                self.meta_mean = np.nanmean(meta_values, axis=0)
                self.meta_std = np.nanstd(meta_values, axis=0)
                self.meta_std[self.meta_std == 0] = 1.0
            else:
                self.meta_mean = np.array([0.0])
                self.meta_std = np.array([1.0])

        self.num_classes = self.df[label_col].nunique()
        logger.info(f"PatentDataset: {len(self)} samples, "
                     f"{self.num_classes} classes, "
                     f"metadata_cols={len(self.metadata_cols)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # --- Text ---
        text = self._get_technology_text(row)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        # --- Metadata ---
        if self.use_metadata and self.metadata_cols:
            meta = row[self.metadata_cols].values.astype(np.float32)
            meta = np.nan_to_num(meta, nan=0.0)
            # Normalize
            meta = (meta - self.meta_mean) / self.meta_std
            item["metadata"] = torch.tensor(meta, dtype=torch.float32)

        # --- Label ---
        label = int(row[self.label_col])
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item

    def _get_technology_text(self, row) -> str:
        """
        Build text representation for a technology from its early patents.
        Concatenates titles (and abstracts if available) from first few patents.
        """
        early_pats = row.get("early_patent_numbers", [])
        if isinstance(early_pats, str):
            try:
                early_pats = json.loads(early_pats)
            except json.JSONDecodeError:
                early_pats = []

        parts = []

        # Add IPC code information as context
        ipc1 = row.get("ipc1", "")
        ipc2 = row.get("ipc2", "")
        if ipc1 and ipc2:
            parts.append(f"Technology components: {ipc1} and {ipc2}.")

        # Add text from early patents (limit to avoid exceeding max_length)
        max_patents_for_text = 3
        for pnum in early_pats[:max_patents_for_text]:
            pat = self.patent_lookup.get(pnum, {})
            title = pat.get("title", "")
            abstract = pat.get("abstract", "")
            if title:
                parts.append(f"Patent {pnum} Title: {title}")
            if abstract:
                parts.append(f"Abstract: {abstract}")

        if not parts:
            # Fallback: use classification code descriptions
            return f"Technology combining CPC codes {ipc1} and {ipc2}."

        return " ".join(parts)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        labels = self.df[self.label_col].values
        class_counts = np.bincount(labels, minlength=self.num_classes)
        total = len(labels)
        weights = total / (self.num_classes * class_counts.astype(np.float32))
        weights = np.nan_to_num(weights, nan=1.0)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler."""
        labels = self.df[self.label_col].values
        class_weights = self.get_class_weights().numpy()
        return class_weights[labels]


# ============================================================
# Data Preparation & Splitting
# ============================================================

def prepare_splits(tech_df: pd.DataFrame, patent_lookup: Dict,
                   cfg: Dict) -> Tuple[PatentDataset, PatentDataset, PatentDataset]:
    """
    Create train/val/test PatentDatasets based on config.

    Supports time-based splitting (recommended) and random splitting.
    """
    split_cfg = cfg["splits"]
    tok_name = cfg["tokenizer"]["name"]
    max_len = cfg["tokenizer"]["max_length"]
    use_meta = cfg["model"]["use_metadata"]

    if split_cfg["method"] == "time_based":
        train_mask = tech_df["emergence_year"] <= split_cfg["train_end_year"]
        val_mask = ((tech_df["emergence_year"] > split_cfg["train_end_year"]) &
                    (tech_df["emergence_year"] <= split_cfg["val_end_year"]))
        test_mask = tech_df["emergence_year"] > split_cfg["val_end_year"]

        train_df = tech_df[train_mask].reset_index(drop=True)
        val_df = tech_df[val_mask].reset_index(drop=True)
        test_df = tech_df[test_mask].reset_index(drop=True)
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        train_val_df, test_df = train_test_split(
            tech_df, test_size=split_cfg["test_ratio"],
            stratify=tech_df["cluster"],
            random_state=split_cfg["random_state"]
        )
        val_ratio_adj = split_cfg["val_ratio"] / (1 - split_cfg["test_ratio"])
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio_adj,
            stratify=train_val_df["cluster"],
            random_state=split_cfg["random_state"]
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, "
                f"test: {len(test_df)}")

    # Log class distribution
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df["cluster"].value_counts().sort_index().to_dict()
        logger.info(f"  {name} class distribution: {dist}")

    train_ds = PatentDataset(train_df, patent_lookup, tok_name, max_len, use_meta)
    val_ds = PatentDataset(val_df, patent_lookup, tok_name, max_len, use_meta)
    test_ds = PatentDataset(test_df, patent_lookup, tok_name, max_len, use_meta)

    # Share normalization stats from training set
    if use_meta:
        val_ds.meta_mean = train_ds.meta_mean
        val_ds.meta_std = train_ds.meta_std
        test_ds.meta_mean = train_ds.meta_mean
        test_ds.meta_std = train_ds.meta_std

    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds: PatentDataset, val_ds: PatentDataset,
                        test_ds: PatentDataset, batch_size: int = 32,
                        use_weighted_sampler: bool = True,
                        num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders with optional weighted sampling for class balance."""

    if use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
