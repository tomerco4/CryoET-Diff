# utils/visualization.py

from typing import Sequence

import os
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt


def central_slices(volume: Tensor | np.ndarray) -> np.ndarray:
    """
    Extract central XY, XZ, YZ slices and return as Hx(3W) numpy array.

    volume: (D,H,W) or (1,D,H,W) or (B,1,D,H,W) -> we use the first.
    """
    if isinstance(volume, Tensor):
        v = volume.detach().cpu().numpy()
    else:
        v = np.asarray(volume)

    # reduce to (D,H,W)
    if v.ndim == 5:
        v = v[0, 0]
    elif v.ndim == 4:
        v = v[0]
    elif v.ndim == 3:
        pass
    else:
        raise ValueError("volume must have 3, 4, or 5 dims.")

    D, H, W = v.shape
    z_c = D // 2
    y_c = H // 2
    x_c = W // 2

    xy = v[z_c]        # (H, W)
    xz = v[:, y_c, :]  # (D, W)
    yz = v[:, :, x_c]  # (D, H) -> we transpose to (H, D) for visualization
    yz = yz.transpose(1, 0)

    # resize (if needed) or just place side-by-side
    # pad to same height
    h_max = max(xy.shape[0], xz.shape[0], yz.shape[0])
    xy_pad = np.pad(xy, ((0, h_max - xy.shape[0]), (0, 0)))
    xz_pad = np.pad(xz, ((0, h_max - xz.shape[0]), (0, 0)))
    yz_pad = np.pad(yz, ((0, h_max - yz.shape[0]), (0, 0)))

    return np.concatenate([xy_pad, xz_pad, yz_pad], axis=1)

def save_volume_slices(volume: Tensor | np.ndarray, path: str, title: str = ""):
    """
    Save a PNG with the three central slices stacked horizontally.
    Includes captions for each slice: XY, XZ, YZ.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get sliced image (Hx3W)
    img = central_slices(volume)
    H, W = img.shape

    # Each slice width
    slice_w = W // 3

    plt.figure(figsize=(9, 3.6))  # slightly taller for captions
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    # Add slice captions
    captions = ["XY", "XZ", "YZ"]
    for i, cap in enumerate(captions):
        x_center = (i + 0.5) * slice_w
        plt.text(
            x_center,          # x position
            H + 5,             # y position (slightly below image)
            cap,
            fontsize=10,
            ha="center",
            va="top",
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round")
        )

    # Title
    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()



def plot_loss_curve(train_losses: Sequence[float],
                    val_losses: Sequence[float] | None,
                    path: str):
    """
    Save a plot of training and (optional) validation MSE loss curves.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(6, 4))

    # Plot training
    plt.plot(train_losses, label="Train MSE", linewidth=2)

    # Plot validation if provided
    if val_losses is not None:
        plt.plot(val_losses, label="Val MSE", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

