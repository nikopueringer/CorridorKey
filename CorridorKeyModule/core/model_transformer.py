from __future__ import annotations

import logging
import sys

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Linear embedding: C_in -> C_out."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DecoderHead(nn.Module):
    def __init__(
        self, feature_channels: list[int] | None = None, embedding_dim: int = 256, output_dim: int = 1
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [112, 224, 448, 896]

        # MLP layers to unify channel dimensions
        self.linear_c4 = MLP(input_dim=feature_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=feature_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=feature_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=feature_channels[0], embed_dim=embedding_dim)

        # Fuse
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

        # Predict
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embedding_dim, output_dim, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features

        n, _, h, w = c4.shape

        # Resize to C1 size (which is H/4)
        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2)).transpose(1, 2).view(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.bn(_c)
        _c = self.relu(_c)

        x = self.dropout(_c)
        x = self.classifier(x)

        return x


class RefinerBlock(nn.Module):
    """
    Residual Block with Dilation and GroupNorm (Safe for Batch Size 2).
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out


class CNNRefinerModule(nn.Module):
    """
    Dilated Residual Refiner (Receptive Field ~65px).
    designed to solve Macroblocking artifacts from Hiera.
    Structure: Stem -> Res(d1) -> Res(d2) -> Res(d4) -> Res(d8) -> Projection.
    """

    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, out_channels: int = 4) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated Residual Blocks (RF Expansion)
        self.res1 = RefinerBlock(hidden_channels, dilation=1)
        self.res2 = RefinerBlock(hidden_channels, dilation=2)
        self.res3 = RefinerBlock(hidden_channels, dilation=4)
        self.res4 = RefinerBlock(hidden_channels, dilation=8)

        # Final Projection (No Activation, purely additive logits)
        self.final = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Tiny Noise Init (Whisper) - Provides gradients without shock
        nn.init.normal_(self.final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.final.bias, 0)

    def forward(self, img: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        # img: [B, 3, H, W]
        # coarse_pred: [B, 4, H, W]
        x = torch.cat([img, coarse_pred], dim=1)

        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # Output Scaling (10x Boost)
        # Allows the Refiner to predict small stable values (e.g. 0.5) that become strong corrections (5.0).
        return self.final(x) * 10.0


# We only tested compilation on windows and linux. For other platforms compilation is disabled as a precaution.
@torch.compile(disable=(sys.platform != "linux" and sys.platform != "win32"))
class GreenFormer(nn.Module):
    def __init__(
        self,
        encoder_name: str = "hiera_base_plus_224.mae_in1k_ft_in1k",
        in_channels: int = 4,
        img_size: int = 512,
        use_refiner: bool = True,
    ) -> None:
        super().__init__()

        # --- Encoder ---
        # Load Pretrained Hiera
        # 1. Create Target Model (512x512, Random Weights)
        # We use features_only=True, which wraps it in FeatureGetterNet
        logger.info("Initializing %s (img_size=%d)", encoder_name, img_size)
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, img_size=img_size)
        # We skip downloading/loading base weights because the user's checkpoint
        # (loaded immediately after this) contains all weights, including correctly
        # trained/sized PosEmbeds. This keeps the project offline-capable using only local assets.
        logger.info("Skipped downloading base weights (relying on custom checkpoint)")

        # Patch First Layer for 4 channels
        if in_channels != 3:
            self._patch_input_layer(in_channels)

        # Get feature info
        # Verified Hiera Base Plus channels: [112, 224, 448, 896]
        # We can try to fetch dynamically
        try:
            feature_channels = self.encoder.feature_info.channels()
        except (AttributeError, TypeError):
            feature_channels = [112, 224, 448, 896]
        logger.info("Feature channels: %s", feature_channels)

        # --- Decoders ---
        embedding_dim = 256

        # Alpha Decoder (Outputs 1 channel)
        self.alpha_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=1)

        # Foreground Decoder (Outputs 3 channels)
        self.fg_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=3)

        # --- Refiner ---
        # CNN Refiner
        # In Channels: 3 (RGB) + 4 (Coarse Pred) = 7
        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = CNNRefinerModule(in_channels=7, hidden_channels=64, out_channels=4)
        else:
            self.refiner = None
            logger.info("Refiner module DISABLED (backbone-only mode)")

    def _patch_input_layer(self, in_channels: int) -> None:
        """
        Modifies the first convolution layer to accept `in_channels`.
        Copies existing RGB weights and initializes extras to zero.
        """
        # Hiera: self.encoder.model.patch_embed.proj

        try:
            patch_embed = self.encoder.model.patch_embed.proj
        except AttributeError:
            # Fallback if timm changes structure or for other models
            patch_embed = self.encoder.patch_embed.proj
        weight = patch_embed.weight.data  # [Out, 3, K, K]
        bias = patch_embed.bias.data if patch_embed.bias is not None else None

        new_in_channels = in_channels
        out_channels, _, k, k = weight.shape

        # Create new conv
        new_conv = nn.Conv2d(
            new_in_channels,
            out_channels,
            kernel_size=k,
            stride=patch_embed.stride,
            padding=patch_embed.padding,
            bias=(bias is not None),
        )

        # Copy weights
        new_conv.weight.data[:, :3, :, :] = weight
        # Initialize new channels to 0 (Weight Patching)
        new_conv.weight.data[:, 3:, :, :] = 0.0

        if bias is not None:
            new_conv.bias.data = bias

        # Replace in module
        try:
            self.encoder.model.patch_embed.proj = new_conv
        except AttributeError:
            self.encoder.patch_embed.proj = new_conv

        logger.info("Patched input layer: 3 → %d channels (extra initialized to 0)", in_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, 4, H, W]
        input_size = x.shape[2:]

        # Encode
        features = self.encoder(x)  # Returns list of features

        # Decode Streams
        alpha_logits = self.alpha_decoder(features)  # [B, 1, H/4, W/4]
        fg_logits = self.fg_decoder(features)  # [B, 3, H/4, W/4]

        # Upsample to full resolution (Bilinear)
        # These are the "Coarse" LOGITS
        alpha_logits_up = F.interpolate(alpha_logits, size=input_size, mode="bilinear", align_corners=False)
        fg_logits_up = F.interpolate(fg_logits, size=input_size, mode="bilinear", align_corners=False)

        # --- HUMILITY CLAMP REMOVED (Phase 3) ---
        # User requested NO CLAMPING to preserve all backbone detail.
        # Refiner sees raw logits (-inf to +inf).
        # alpha_logits_up = torch.clamp(alpha_logits_up, -3.0, 3.0)
        # fg_logits_up = torch.clamp(fg_logits_up, -3.0, 3.0)

        # Coarse Probs (for Loss and Refiner Input)
        alpha_coarse = torch.sigmoid(alpha_logits_up)
        fg_coarse = torch.sigmoid(fg_logits_up)

        # --- Refinement (CNN Hybrid) ---
        # 4. Refine (CNN)
        # Input to refiner: RGB Image (first 3 channels of x) + Coarse Predictions (Probs)
        # We give the refiner 'Probs' as input features because they are normalized [0,1]
        rgb = x[:, :3, :, :]

        # Feed the Refiner
        coarse_pred = torch.cat([alpha_coarse, fg_coarse], dim=1)  # [B, 4, H, W]

        # Refiner outputs DELTA LOGITS
        # The refiner predicts the correction in valid score space (-inf, inf)
        if self.use_refiner and self.refiner is not None:
            delta_logits = self.refiner(rgb, coarse_pred)
        else:
            # Zero Deltas
            delta_logits = torch.zeros_like(coarse_pred)

        delta_alpha = delta_logits[:, 0:1]
        delta_fg = delta_logits[:, 1:4]

        # Residual Addition in Logit Space
        # This allows infinite correction capability and prevents saturation blocking
        alpha_final_logits = alpha_logits_up + delta_alpha
        fg_final_logits = fg_logits_up + delta_fg

        # Final Activation
        alpha_final = torch.sigmoid(alpha_final_logits)
        fg_final = torch.sigmoid(fg_final_logits)

        return {"alpha": alpha_final, "fg": fg_final}
