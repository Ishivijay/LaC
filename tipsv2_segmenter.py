#!/usr/bin/env python3
"""TIPSv2 Zero-Shot Segmentation for Walkable Area Detection
============================================================

Uses TIPSv2 (Text-Image Pretraining with Spatial Awareness) from Google DeepMind
for zero-shot semantic segmentation of walkable areas in indoor scenes.

The approach:
  1. Encode class-specific text prompts (e.g., "floor", "stairs", "wall")
     using the TIPSv2 text encoder
  2. Encode image patches using the TIPSv2 vision encoder (ViT with
     value-attention trick from MaskCLIP)
  3. Compute cosine similarity between patch features and text features
  4. Argmax over classes → segmentation map
  5. Extract walkable class mask (union of walkable classes)

Supports two inference modes:
  - "whole":  Single forward pass over the whole image (fast, lower res)
  - "slide":  Sliding window inference (slower, higher quality)

Usage:
    from tipsv2_segmenter import TIPSv2Segmenter

    seg = TIPSv2Segmenter(
        variant="L",
        checkpoint_dir="/path/to/checkpoints",
        device="cuda",
    )
    mask = seg.segment_walkable(pil_image)
    # mask is a boolean numpy array (H, W), True = walkable

References:
    - TIPSv2: https://arxiv.org/abs/2604.12012
    - TIPSv1: https://arxiv.org/abs/2410.16512
    - Code:   https://github.com/google-deepmind/tips
"""

import io
import logging
import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from PIL import Image
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_SIZE = 14
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 64
IMAGE_SIZE = 448  # Default input size for TIPSv2

# Normalization: TIPS uses identity normalization (mean=0, std=1)
NORMALIZE_TIPS = TVT.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

# Checkpoint download URLs
CHECKPOINT_BASE_URL = "https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch"
TOKENIZER_URL = "https://storage.googleapis.com/tips_data/v1_0/checkpoints/tokenizer.model"

V2_CHECKPOINT_MAP = {
    "B": ("tips_v2_oss_b14_vision.npz", "tips_v2_oss_b14_text.npz"),
    "L": ("tips_v2_oss_l14_vision.npz", "tips_v2_oss_l14_text.npz"),
    "So": ("tips_v2_oss_so14_vision.npz", "tips_v2_oss_so14_text.npz"),
    "g": ("tips_v2_oss_g14_vision.npz", "tips_v2_oss_g14_text.npz"),
}

# Text encoder configs per variant (same for v1 and v2)
TEXT_CONFIGS = {
    "B": {"hidden_size": 768, "mlp_dim": 3072, "num_heads": 12, "num_layers": 12},
    "L": {"hidden_size": 1024, "mlp_dim": 4096, "num_heads": 16, "num_layers": 12},
    "So": {"hidden_size": 1152, "mlp_dim": 4304, "num_heads": 16, "num_layers": 27},
    "g": {"hidden_size": 1536, "mlp_dim": 6144, "num_heads": 24, "num_layers": 12},
}

# Prompt templates (from TCL paper, used in TIPS zero-shot segmentation)
PROMPT_TEMPLATES = [
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
    "a photo of many {}.",
    "a photo of {}s.",
]

# ---------------------------------------------------------------------------
# Walkable area class definitions
# ---------------------------------------------------------------------------

# Classes that represent walkable surfaces
WALKABLE_CLASSES = [
    "floor",
    "stairs",
    "stairway",
    "ramp",
    "path",
    "sidewalk",
    "runway",
    "step",
    "corridor",
]

# Classes that represent non-walkable surfaces (background)
NON_WALKABLE_CLASSES = [
    "wall",
    "ceiling",
    "door",
    "windowpane",
    "furniture",
    "table",
    "chair",
    "sofa",
    "bed",
    "cabinet",
    "shelf",
    "desk",
    "counter",
    "column",
    "pillar",
    "railing",
    "fence",
    "curtain",
    "painting",
    "mirror",
    "lamp",
    "plant",
    "tree",
    "grass",
    "sky",
    "building",
    "road",
    "person",
    "car",
    "obstacle",
    "box",
    "bag",
    "rock",
]

# Default: binary classification (walkable vs non-walkable)
DEFAULT_CLASSES = {
    "walkable": WALKABLE_CLASSES,
    "non_walkable": NON_WALKABLE_CLASSES,
}


# ---------------------------------------------------------------------------
# Helper: Short-side resize
# ---------------------------------------------------------------------------

class ShortSideResize(nn.Module):
    """Resize image so the short side equals `size`, preserving aspect ratio."""

    def __init__(self, size: int, interpolation: TVT.InterpolationMode) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        _, h, w = TVTF.get_dimensions(img)
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            new_w = self.size
            new_h = int(self.size * h / w)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)
        else:
            new_h = self.size
            new_w = int(self.size * w / h)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)


# ---------------------------------------------------------------------------
# Vision encoder helpers (value-attention trick from MaskCLIP)
# ---------------------------------------------------------------------------

def _get_all_blocks(model_image):
    """Get all transformer blocks from the ViT model."""
    if model_image.chunked_blocks:
        blocks = []
        for chunk in model_image.blocks:
            for blk in chunk:
                if not isinstance(blk, nn.Identity):
                    blocks.append(blk)
        return blocks
    return list(model_image.blocks)


def encode_image_value_attention(model_image, img: Tensor) -> Tensor:
    """Encode image using the value-attention trick from MaskCLIP.

    Instead of using the final patch embeddings, we use the value projections
    from the last attention layer. This produces better spatial features for
    dense prediction tasks like segmentation.

    Args:
        model_image: TIPSv2 vision encoder (ViT).
        img: Input image tensor [B, C, H, W].

    Returns:
        Patch features [B, h, w, D] where h=H/patch_size, w=W/patch_size.
    """
    B, _, H, W = img.shape
    P = model_image.patch_size
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P

    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)

    B, _, h_i, w_i = img.shape

    x = model_image.prepare_tokens_with_masks(img)

    num_register = model_image.num_register_tokens
    all_blocks = _get_all_blocks(model_image)
    for i, blk in enumerate(all_blocks):
        if i < len(all_blocks) - 1:
            x = blk(x)
        else:
            # Last block: extract value attention features
            x_normed = blk.norm1(x)
            b_dim, n_dim, c_dim = x_normed.shape
            qkv = (
                blk.attn.qkv(x_normed)
                .reshape(b_dim, n_dim, 3, blk.attn.num_heads, c_dim // blk.attn.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            v = qkv[2]
            v_out = v.transpose(1, 2).reshape(b_dim, n_dim, c_dim)
            v_out = blk.attn.proj(v_out)
            v_out = blk.ls1(v_out)
            x_val = v_out + x

            y_val = blk.norm2(x_val)
            y_val = blk.ls2(blk.mlp(y_val))
            x_val = x_val + y_val

    x_val = model_image.norm(x_val)
    patch_tokens = x_val[:, 1 + num_register:, :]
    blocks_patches = patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    return blocks_patches


# ---------------------------------------------------------------------------
# Segmentation inference functions
# ---------------------------------------------------------------------------

def predict_whole(model_image, img: Tensor, text_features: Tensor) -> Tensor:
    """Predict segmentation for a whole image.

    Args:
        model_image: TIPSv2 vision encoder.
        img: Image tensor [C, H, W].
        text_features: Text features [num_classes, D].

    Returns:
        Cosine similarity map [num_classes, h, w].
    """
    blocks_feats = encode_image_value_attention(model_image, img.unsqueeze(0))
    _, h, w, _ = blocks_feats.shape
    blocks_feats = blocks_feats.squeeze(0)

    blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)
    cos = torch.einsum("cd,hwd->chw", text_features, blocks_feats)

    return cos


def predict_slide(
    model_image, img: Tensor, text_features: Tensor,
    side: int, stride: int,
) -> Tensor:
    """Predict segmentation using sliding window inference.

    Follows TCL approach for higher-quality segmentation on large images.

    Args:
        model_image: TIPSv2 vision encoder.
        img: Image tensor [C, H, W].
        text_features: Text features [num_classes, D].
        side: Window size (typically image_size, e.g., 448).
        stride: Stride for sliding window.

    Returns:
        Probability map [num_classes, H, W].
    """
    _, H, W = img.shape
    num_classes, _ = text_features.shape
    device = img.device
    probs = torch.zeros([num_classes, H, W], device=device)
    counts = torch.zeros([H, W], device=device)

    h_grids = max(H - side + stride - 1, 0) // stride + 1
    w_grids = max(W - side + stride - 1, 0) // stride + 1

    for i in range(h_grids):
        for j in range(w_grids):
            y1 = i * stride
            x1 = j * stride
            y2 = min(y1 + side, H)
            x2 = min(x1 + side, W)
            y1 = max(y2 - side, 0)
            x1 = max(x2 - side, 0)

            img_window = img[:, y1:y2, x1:x2]
            cos = predict_whole(model_image, img_window, text_features)

            cos = F.interpolate(
                cos.unsqueeze(0),
                size=img_window.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            probs[:, y1:y2, x1:x2] += cos.softmax(dim=0)
            counts[y1:y2, x1:x2] += 1

    probs /= counts
    return probs


# ---------------------------------------------------------------------------
# SentencePiece tokenizer (replaces TF-based tokenizer from TIPS repo)
# ---------------------------------------------------------------------------

class _SPPretrainedTokenizer:
    """SentencePiece-based tokenizer compatible with TIPS text encoder.

    Replaces the tensorflow_text-based Tokenizer from the TIPS repo.
    Produces the same output format: (token_ids, paddings) as numpy arrays.
    """

    def __init__(self, tokenizer_path: str):
        """Load SentencePiece model.

        Args:
            tokenizer_path: Path to the tokenizer.model file.
        """
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(tokenizer_path)
        logger.info(f"Loaded SentencePiece tokenizer from {tokenizer_path}")

    def tokenize(self, texts: List[str], max_len: int = 64):
        """Tokenize texts with padding.

        Args:
            texts: List of strings to tokenize.
            max_len: Maximum sequence length.

        Returns:
            Tuple of (token_ids, paddings) as numpy arrays.
            token_ids: int64 array [batch, max_len]
            paddings: float32 array [batch, max_len] (1.0 = padding, 0.0 = token)
        """
        batch_size = len(texts)
        token_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        paddings = np.ones((batch_size, max_len), dtype=np.float32)

        for i, text in enumerate(texts):
            ids = self._sp.EncodeAsIds(text.lower())
            length = min(len(ids), max_len)
            token_ids[i, :length] = ids[:length]
            paddings[i, :length] = 0.0

        return token_ids, paddings


# ---------------------------------------------------------------------------
# TIPSv2 Segmenter class
# ---------------------------------------------------------------------------

class TIPSv2Segmenter:
    """TIPSv2-based zero-shot segmentation for walkable area detection.

    This class encapsulates the TIPSv2 vision and text encoders and provides
    a simple API for segmenting walkable areas in images.

    Example:
        seg = TIPSv2Segmenter(variant="L", checkpoint_dir="/path/to/ckpts")
        mask = seg.segment_walkable(pil_image)
        # mask: boolean numpy array (H, W), True = walkable
    """

    def __init__(
        self,
        variant: str = "L",
        checkpoint_dir: str = None,
        device: str = "cuda",
        image_size: int = IMAGE_SIZE,
        inference_mode: str = "slide",
        stride: int = 336,
        walkable_classes: Optional[Dict[str, List[str]]] = None,
        tips_repo_path: Optional[str] = None,
    ):
        """Initialize TIPSv2 segmenter.

        Args:
            variant: Model variant — "B", "L", "So", or "g".
            checkpoint_dir: Directory containing TIPSv2 checkpoints.
                If None, uses $WORK/tipsv2_checkpoints.
            device: Device for inference ("cuda" or "cpu").
            image_size: Input image size (default: 448).
            inference_mode: "whole" (fast) or "slide" (higher quality).
            stride: Stride for sliding window inference (only used in "slide" mode).
            walkable_classes: Dict of {"walkable": [...], "non_walkable": [...]}.
                If None, uses DEFAULT_CLASSES.
            tips_repo_path: Path to the cloned TIPS repository.
                If None, auto-detects from $HOME/tips_repo or workspace.
        """
        self.variant = variant
        self.device = device
        self.image_size = image_size
        self.inference_mode = inference_mode
        self.stride = stride
        self.walkable_classes = walkable_classes or DEFAULT_CLASSES

        # Resolve paths
        self.checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)
        self.tips_repo_path = self._resolve_tips_repo(tips_repo_path)

        # Add TIPS repo to sys.path
        import sys
        tips_parent = str(Path(self.tips_repo_path).parent)
        if tips_parent not in sys.path:
            sys.path.insert(0, tips_parent)

        # Download checkpoints if needed
        self._ensure_checkpoints()

        # Load models
        self.model_image = None
        self.model_text = None
        self.tokenizer = None
        self.text_features = None
        self._load_models()
        self._encode_text_features()

        logger.info(f"TIPSv2Segmenter initialized: variant={variant}, "
                     f"device={device}, mode={inference_mode}")

    @staticmethod
    def _resolve_checkpoint_dir(checkpoint_dir: Optional[str]) -> Path:
        """Resolve checkpoint directory."""
        if checkpoint_dir:
            return Path(checkpoint_dir)
        work = os.environ.get("WORK", str(Path.home()))
        return Path(work) / "tipsv2_checkpoints"

    @staticmethod
    def _resolve_tips_repo(tips_repo_path: Optional[str]) -> Path:
        """Resolve TIPS repository path.

        The repo must be importable as `tips` package. If cloned as `tips_repo`,
        a symlink `tips -> tips_repo` is created automatically.
        """
        if tips_repo_path:
            return Path(tips_repo_path)

        # Try common locations — prefer 'tips' (correct package name) over 'tips_repo'
        candidates = [
            Path.home() / "tips",
            Path.home() / "tips_repo",
            Path(__file__).resolve().parent.parent / "tips",
            Path(__file__).resolve().parent.parent / "tips_repo",
        ]
        for candidate in candidates:
            if (candidate / "pytorch" / "image_encoder.py").exists():
                # Ensure a 'tips' symlink exists for Python imports
                tips_link = candidate.parent / "tips"
                if candidate.name == "tips_repo" and not tips_link.exists():
                    try:
                        os.symlink(str(candidate), str(tips_link))
                        logger.info(f"Created symlink: {tips_link} -> {candidate}")
                    except OSError:
                        pass
                return candidate
        raise FileNotFoundError(
            "TIPS repository not found. Clone it with:\n"
            "  git clone https://github.com/google-deepmind/tips.git ~/tips\n"
            "Or:\n"
            "  git clone https://github.com/google-deepmind/tips.git ~/tips_repo\n"
            f"Searched: {[str(c) for c in candidates]}"
        )

    def _ensure_checkpoints(self):
        """Download checkpoints if they don't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        vision_name, text_name = V2_CHECKPOINT_MAP[self.variant]

        for ckpt_name in [vision_name, text_name]:
            ckpt_path = self.checkpoint_dir / ckpt_name
            if not ckpt_path.exists():
                logger.info(f"Downloading TIPSv2 checkpoint: {ckpt_name}...")
                urllib.request.urlretrieve(
                    f"{CHECKPOINT_BASE_URL}/{ckpt_name}", str(ckpt_path)
                )
                logger.info(f"  Saved to {ckpt_path}")

        # Download tokenizer
        tokenizer_path = self.checkpoint_dir / "tokenizer.model"
        if not tokenizer_path.exists():
            logger.info("Downloading TIPSv2 tokenizer...")
            urllib.request.urlretrieve(TOKENIZER_URL, str(tokenizer_path))
            logger.info(f"  Saved to {tokenizer_path}")

    def _load_models(self):
        """Load TIPSv2 vision and text encoders.

        Handles the tensorflow dependency gracefully: if tensorflow is not
        available, mocks it so the text_encoder module can still be imported
        (only the TextEncoder PyTorch class is needed, not the TF tokenizer).
        Uses sentencepiece directly for tokenization instead.
        """
        import importlib
        import sys

        # Disable xFormers on CPU (it only supports CUDA)
        if self.device == "cpu":
            os.environ["XFORMERS_DISABLED"] = "1"

        tips_parent = str(Path(self.tips_repo_path).parent)
        if tips_parent not in sys.path:
            sys.path.insert(0, tips_parent)

        # Import vision encoder (pure PyTorch, no TF dependency)
        from tips.pytorch import image_encoder

        # Import text encoder — mock tensorflow if not available
        text_encoder = None
        try:
            import tensorflow  # noqa: F401
            import tensorflow_text  # noqa: F401
            from tips.pytorch import text_encoder
        except ImportError:
            logger.info("tensorflow not available — using direct import for text_encoder")
            # Import text_encoder directly by reading and executing its source,
            # bypassing the tensorflow imports at module level.
            import importlib.util
            te_path = Path(self.tips_repo_path) / "pytorch" / "text_encoder.py"
            spec = importlib.util.spec_from_file_location("text_encoder", str(te_path))
            text_encoder = importlib.util.module_from_spec(spec)
            # Pre-populate sys.modules with mocks for tensorflow dependencies
            import types
            for mod_name in ["tensorflow", "tensorflow_text"]:
                if mod_name not in sys.modules:
                    mock_mod = types.ModuleType(mod_name)
                    mock_mod.__spec__ = type('ModuleSpec', (), {
                        'name': mod_name, 'loader': None, 'origin': None
                    })()
                    mock_mod.__path__ = []
                    mock_mod.__file__ = None
                    sys.modules[mod_name] = mock_mod
            spec.loader.exec_module(text_encoder)

        vision_name, text_name = V2_CHECKPOINT_MAP[self.variant]
        vision_path = self.checkpoint_dir / vision_name
        text_path = self.checkpoint_dir / text_name
        tokenizer_path = self.checkpoint_dir / "tokenizer.model"

        # ── Vision encoder ──
        logger.info(f"Loading TIPSv2 vision encoder: {vision_name}")
        weights_image = dict(np.load(str(vision_path), allow_pickle=False))
        for key in weights_image:
            weights_image[key] = torch.tensor(weights_image[key])

        ffn_layer = "swiglu" if self.variant == "g" else "mlp"

        # Get model constructor
        model_constructors = {
            "B": image_encoder.vit_base,
            "L": image_encoder.vit_large,
            "So": image_encoder.vit_so400m,
            "g": image_encoder.vit_giant2,
        }
        model_fn = model_constructors[self.variant]

        with torch.no_grad():
            self.model_image = model_fn(
                img_size=self.image_size,
                patch_size=PATCH_SIZE,
                ffn_layer=ffn_layer,
                block_chunks=0,
                init_values=1.0,
                interpolate_antialias=True,
                interpolate_offset=0.0,
            )
            self.model_image.load_state_dict(weights_image)
            self.model_image = self.model_image.to(self.device).eval()

        # ── Text encoder ──
        logger.info(f"Loading TIPSv2 text encoder: {text_name}")
        with open(text_path, "rb") as fin:
            inbuffer = io.BytesIO(fin.read())
        np_weights_text = dict(np.load(inbuffer, allow_pickle=False))

        weights_text = {}
        for key, value in np_weights_text.items():
            weights_text[key] = torch.from_numpy(value)

        # Pop non-model keys
        weights_text.pop("temperature", None)
        weights_text.pop("temperature_contrastive", None)

        text_config = TEXT_CONFIGS[self.variant]

        with torch.no_grad():
            self.model_text = text_encoder.TextEncoder(
                text_config,
                vocab_size=VOCAB_SIZE,
            )
            self.model_text.load_state_dict(weights_text)
            self.model_text = self.model_text.to(self.device).eval()

        # ── Tokenizer (using sentencepiece directly) ──
        logger.info("Loading TIPSv2 tokenizer (sentencepiece)...")
        self.tokenizer = _SPPretrainedTokenizer(str(tokenizer_path))

        logger.info("TIPSv2 models loaded successfully")

    def _encode_text(self, texts: List[str]) -> Tensor:
        """Encode a list of text strings into normalized features.

        Args:
            texts: List of text strings.

        Returns:
            Normalized text features [D].
        """
        token_ids, paddings = self.tokenizer.tokenize(texts, max_len=MAX_SEQ_LEN)
        ids_t = torch.from_numpy(token_ids).to(self.device)
        pads_t = torch.from_numpy(paddings).to(self.device)

        with torch.no_grad():
            feats = self.model_text(ids_t, pads_t)
            feats = F.normalize(feats, p=2, dim=-1)
            feats = feats.mean(dim=0)
            feats = F.normalize(feats, p=2, dim=-1)

        return feats.float()

    def _encode_text_features(self):
        """Pre-compute text features for all classes using prompt templates.

        For binary segmentation (walkable vs non-walkable), we encode each
        sub-class with multiple prompt templates and average the features.
        """
        logger.info("Encoding text features for walkable area classes...")
        text_feats = []

        self._class_names = []
        for group_name, sub_classes in self.walkable_classes.items():
            self._class_names.append(group_name)
            # Encode all sub-classes with prompt templates, then average
            group_feats = []
            for sub_class in sub_classes:
                texts = [template.format(sub_class) for template in PROMPT_TEMPLATES]
                feat = self._encode_text(texts)
                group_feats.append(feat)

            # Average over sub-classes
            group_feat = torch.stack(group_feats).mean(dim=0)
            group_feat = F.normalize(group_feat, p=2, dim=-1)
            text_feats.append(group_feat)

        self.text_features = torch.stack(text_feats).to(self.device)
        logger.info(f"Text features shape: {self.text_features.shape}")
        logger.info(f"Classes: {self._class_names}")

    def _preprocess_image(self, pil_image: Image.Image) -> Tensor:
        """Preprocess a PIL image for TIPSv2 inference.

        Resizes short side to self.image_size, converts to tensor,
        and applies TIPS normalization.

        Args:
            pil_image: Input PIL image.

        Returns:
            Preprocessed tensor [C, H, W].
        """
        transform = TVT.Compose([
            ShortSideResize(self.image_size, TVT.InterpolationMode.BICUBIC),
            TVT.ToTensor(),
            NORMALIZE_TIPS,
        ])
        return transform(pil_image)

    def segment(
        self,
        pil_image: Image.Image,
        return_prob_map: bool = False,
    ) -> Dict:
        """Run zero-shot segmentation on an image.

        Args:
            pil_image: Input PIL image (RGB).
            return_prob_map: If True, also return the probability map.

        Returns:
            Dict with:
                - "segmentation": numpy array (H, W) with class indices
                - "walkable_mask": boolean numpy array (H, W)
                - "prob_map": (optional) float numpy array (num_classes, H, W)
                - "class_names": list of class names
        """
        img_tensor = self._preprocess_image(pil_image)
        _, H, W = img_tensor.shape

        with torch.inference_mode():
            if self.inference_mode == "whole":
                cos_map = predict_whole(self.model_image, img_tensor, self.text_features)
                prob_map = cos_map.softmax(dim=0)
            elif self.inference_mode == "slide":
                prob_map = predict_slide(
                    self.model_image, img_tensor, self.text_features,
                    self.image_size, self.stride,
                )
            else:
                raise ValueError(f"Unknown inference mode: {self.inference_mode}")

        # Upsample to original image size
        prob_map_up = F.interpolate(
            prob_map.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Segmentation: argmax over classes
        seg = prob_map_up.argmax(dim=0).cpu().numpy()

        # Walkable mask: class index 0 = "walkable"
        walkable_idx = self._class_names.index("walkable")
        walkable_mask = (seg == walkable_idx)

        result = {
            "segmentation": seg,
            "walkable_mask": walkable_mask,
            "class_names": self._class_names,
        }

        if return_prob_map:
            result["prob_map"] = prob_map_up.cpu().numpy()

        return result

    def segment_walkable(self, pil_image: Image.Image) -> np.ndarray:
        """Segment walkable areas in an image.

        Convenience method that returns just the walkable mask.

        Args:
            pil_image: Input PIL image (RGB).

        Returns:
            Boolean numpy array (H, W), True = walkable.
        """
        result = self.segment(pil_image)
        return result["walkable_mask"]

    def segment_with_components(
        self,
        pil_image: Image.Image,
        min_area_pixels: int = 100,
    ) -> Dict:
        """Segment walkable areas and extract connected components.

        This is the main method for integration with the walkable area pipeline.
        It returns individual walkable regions as separate masks, similar to
        how SA2VA returns multiple masks.

        Args:
            pil_image: Input PIL image (RGB).
            min_area_pixels: Minimum area (in pixels) for a component to be kept.

        Returns:
            Dict with:
                - "walkable_mask": boolean numpy array (H, W)
                - "components": list of dicts with "mask", "bbox", "area_pixels"
                - "prob_map": float numpy array (num_classes, H, W)
                - "segmentation": numpy array (H, W)
                - "class_names": list of class names
        """
        result = self.segment(pil_image, return_prob_map=True)
        walkable_mask = result["walkable_mask"]
        prob_map = result["prob_map"]

        # Extract connected components
        try:
            import cv2
            walkable_uint8 = walkable_mask.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(walkable_uint8)

            components = []
            for label_id in range(1, num_labels):  # Skip background (0)
                comp_mask = labels == label_id
                area = int(comp_mask.sum())
                if area < min_area_pixels:
                    continue

                ys, xs = np.where(comp_mask)
                bbox = {
                    "x1": int(xs.min()),
                    "y1": int(ys.min()),
                    "x2": int(xs.max()) + 1,
                    "y2": int(ys.max()) + 1,
                }

                components.append({
                    "mask": comp_mask,
                    "bbox": bbox,
                    "area_pixels": area,
                })
        except ImportError:
            # Fallback without OpenCV: treat entire walkable area as one component
            logger.warning("OpenCV not available — treating entire walkable area as one component")
            area = int(walkable_mask.sum())
            if area >= min_area_pixels:
                ys, xs = np.where(walkable_mask)
                bbox = {
                    "x1": int(xs.min()),
                    "y1": int(ys.min()),
                    "x2": int(xs.max()) + 1,
                    "y2": int(ys.max()) + 1,
                }
                components = [{
                    "mask": walkable_mask,
                    "bbox": bbox,
                    "area_pixels": area,
                }]

        result["components"] = components
        return result


# ---------------------------------------------------------------------------
# Standalone inference function (for quick testing)
# ---------------------------------------------------------------------------

def run_tipsv2_inference(
    image_path: str,
    variant: str = "L",
    checkpoint_dir: str = None,
    device: str = "cuda",
    inference_mode: str = "slide",
    output_path: str = None,
):
    """Run TIPSv2 walkable area segmentation on a single image.

    Args:
        image_path: Path to input image.
        variant: Model variant ("B", "L", "So", "g").
        checkpoint_dir: Path to checkpoints directory.
        device: Device ("cuda" or "cpu").
        inference_mode: "whole" or "slide".
        output_path: Path to save visualization (optional).
    """
    pil_image = Image.open(image_path).convert("RGB")
    logger.info(f"Image size: {pil_image.size}")

    seg = TIPSv2Segmenter(
        variant=variant,
        checkpoint_dir=checkpoint_dir,
        device=device,
        inference_mode=inference_mode,
    )

    result = seg.segment_with_components(pil_image)

    logger.info(f"Walkable pixels: {result['walkable_mask'].sum()}")
    logger.info(f"Connected components: {len(result['components'])}")

    for i, comp in enumerate(result["components"]):
        logger.info(f"  Component {i}: {comp['area_pixels']} pixels, bbox={comp['bbox']}")

    if output_path:
        # Save visualization
        rgb = np.array(pil_image)
        overlay = rgb.copy()
        walkable_color = np.zeros_like(rgb)
        walkable_color[result["walkable_mask"]] = [0, 255, 0]
        overlay = (overlay * 0.5 + walkable_color * 0.5).astype(np.uint8)

        vis = np.concatenate([rgb, overlay], axis=1)
        Image.fromarray(vis).save(output_path)
        logger.info(f"Visualization saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="TIPSv2 Walkable Area Segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--variant", default="L", choices=["B", "L", "So", "g"],
                        help="Model variant (default: L)")
    parser.add_argument("--checkpoint_dir", default=None, help="Checkpoint directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--mode", default="slide", choices=["whole", "slide"],
                        help="Inference mode (default: slide)")
    parser.add_argument("--output", default=None, help="Output visualization path")

    args = parser.parse_args()
    run_tipsv2_inference(
        image_path=args.image,
        variant=args.variant,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        inference_mode=args.mode,
        output_path=args.output,
    )
