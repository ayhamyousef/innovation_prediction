"""
Transformer-based patent classifier for innovation growth curve prediction.

Supports multiple fusion strategies for combining text embeddings with
numerical metadata features:
  - concat: simple concatenation
  - gated: learned gating mechanism
  - cross_attention: cross-attention between text and metadata
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

logger = logging.getLogger("innovation_prediction.model")


class MetadataEncoder(nn.Module):
    """Encode numerical metadata features into a dense representation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    """Gated fusion of text and metadata embeddings."""

    def __init__(self, text_dim: int, meta_dim: int, output_dim: int):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.meta_proj = nn.Linear(meta_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(text_dim + meta_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, text_emb: torch.Tensor,
                meta_emb: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([text_emb, meta_emb], dim=-1))
        text_proj = self.text_proj(text_emb)
        meta_proj = self.meta_proj(meta_emb)
        return g * text_proj + (1 - g) * meta_proj


class CrossAttentionFusion(nn.Module):
    """Cross-attention between text [CLS] token and metadata."""

    def __init__(self, text_dim: int, meta_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, text_dim)
        self.meta_proj = nn.Linear(meta_dim, text_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, text_emb: torch.Tensor,
                meta_emb: torch.Tensor) -> torch.Tensor:
        # text_emb: (B, text_dim), meta_emb: (B, meta_dim)
        q = self.text_proj(text_emb).unsqueeze(1)   # (B, 1, D)
        kv = self.meta_proj(meta_emb).unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.attn(q, kv, kv)          # (B, 1, D)
        out = self.norm(q + attn_out).squeeze(1)     # (B, D)
        return out


class PatentTransformerClassifier(nn.Module):
    """
    Transformer encoder + metadata fusion for 4-class patent classification.

    Architecture:
      1. BERT-like backbone encodes patent text → [CLS] embedding
      2. MetadataEncoder encodes 7 numerical features → dense vector
      3. Fusion layer combines text + metadata
      4. Classification head outputs logits for 4 growth curves
    """

    def __init__(self, backbone: str = "bert-base-uncased",
                 num_classes: int = 4,
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 use_metadata: bool = True,
                 metadata_features_count: int = 7,
                 metadata_dim: int = 32,
                 fusion_strategy: str = "concat",
                 freeze_backbone_layers: int = 0):
        super().__init__()

        self.use_metadata = use_metadata
        self.fusion_strategy = fusion_strategy

        # Text encoder
        self.backbone_config = AutoConfig.from_pretrained(backbone)
        self.backbone = AutoModel.from_pretrained(backbone)
        text_dim = self.backbone_config.hidden_size

        # Optionally freeze early layers
        if freeze_backbone_layers > 0:
            self._freeze_layers(freeze_backbone_layers)

        # Metadata encoder
        if use_metadata and metadata_features_count > 0:
            self.metadata_encoder = MetadataEncoder(
                metadata_features_count, metadata_dim, dropout
            )
        else:
            self.metadata_encoder = None

        # Fusion
        if use_metadata and metadata_features_count > 0:
            if fusion_strategy == "concat":
                classifier_input_dim = text_dim + metadata_dim
            elif fusion_strategy == "gated":
                self.fusion = GatedFusion(text_dim, metadata_dim, hidden_dim)
                classifier_input_dim = hidden_dim
            elif fusion_strategy == "cross_attention":
                self.fusion = CrossAttentionFusion(text_dim, metadata_dim)
                classifier_input_dim = text_dim
            else:
                raise ValueError(f"Unknown fusion: {fusion_strategy}")
        else:
            classifier_input_dim = text_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        logger.info(f"PatentTransformerClassifier initialized: "
                     f"backbone={backbone}, fusion={fusion_strategy}, "
                     f"use_metadata={use_metadata}, "
                     f"classifier_input_dim={classifier_input_dim}")

    def _freeze_layers(self, n_layers: int):
        """Freeze the first n layers of the backbone."""
        # Freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        # Freeze encoder layers
        if hasattr(self.backbone, "encoder"):
            for i, layer in enumerate(self.backbone.encoder.layer):
                if i < n_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        logger.info(f"Froze {n_layers} backbone layers + embeddings")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                metadata: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with 'logits' and optionally 'loss'.
        """
        # Encode text
        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**backbone_kwargs)
        # Use [CLS] token representation
        text_emb = outputs.last_hidden_state[:, 0, :]  # (B, text_dim)

        # Fuse with metadata
        if self.use_metadata and metadata is not None and self.metadata_encoder is not None:
            meta_emb = self.metadata_encoder(metadata)  # (B, meta_dim)

            if self.fusion_strategy == "concat":
                fused = torch.cat([text_emb, meta_emb], dim=-1)
            elif self.fusion_strategy in ("gated", "cross_attention"):
                fused = self.fusion(text_emb, meta_emb)
            else:
                fused = text_emb
        else:
            fused = text_emb

        # Classify
        logits = self.classifier(fused)  # (B, num_classes)

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result


def build_model(cfg: Dict) -> PatentTransformerClassifier:
    """Build model from config dict."""
    model_cfg = cfg["model"]
    return PatentTransformerClassifier(
        backbone=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        use_metadata=model_cfg["use_metadata"],
        metadata_features_count=model_cfg["metadata_features_count"],
        metadata_dim=model_cfg["metadata_dim"],
        fusion_strategy=model_cfg["fusion_strategy"],
        freeze_backbone_layers=model_cfg["freeze_backbone_layers"],
    )
