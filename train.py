# train.py

import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from data.particles_dataset import CryoETParticlesDataset
from models.diffusion_unet3d import DiffusionUNet3D
from diffusion.schedule import DiffusionSchedule
from utils.logger import CSVLogger
from utils.visualization import plot_loss_curve, save_volume_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D conditional DDPM for CryoET particle generation"
    )
    parser.add_argument("--train_tomo_dir", type=str, required=True,
                        help="Path to train tomograms directory (with .mrc files).")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to particles_all_bin0.csv.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Cubic subvolume size (default 64).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=1000,
                        help="Number of diffusion steps.")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--schedule", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of data to use as validation.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Checkpoint interval (for saving checkpoints and logs).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset: we use all experiments in train_tomo_dir; we don't split by experiment name here.
    dataset = CryoETParticlesDataset(
        tomogram_root=args.train_tomo_dir,
        csv_path=args.csv_path,
        patch_size=args.patch_size,
        cache_volumes=True,
    )

    # Simple random train/val split by index
    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model, schedule, optimizer
    model = DiffusionUNet3D(num_classes=dataset.num_classes).to(device)
    schedule = DiffusionSchedule(
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger = CSVLogger(args.output_dir)
    epoch_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (patches, class_vec, coords) in enumerate(train_loader):
            patches = patches.to(device)        # (B,1,D,H,W)
            class_vec = class_vec.to(device)    # (B,C)
            coords = coords.to(device)          # (B,3)

            B = patches.size(0)
            # timesteps: 0..T-1
            t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)

            # sample noise
            noise = torch.randn_like(patches)

            # q(x_t | x_0)
            x_noisy = schedule.q_sample(patches, t, noise=noise)

            # network predicts x0 (clean)
            x_pred = model(x_noisy, t, class_vec, coords)

            loss = F.mse_loss(x_pred, patches)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B

        train_loss = running_loss / len(train_set)

        # simple validation MSE
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, class_vec, coords in val_loader:
                patches = patches.to(device)
                class_vec = class_vec.to(device)
                coords = coords.to(device)
                B = patches.size(0)
                t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)
                noise = torch.randn_like(patches)
                x_noisy = schedule.q_sample(patches, t, noise=noise)
                x_pred = model(x_noisy, t, class_vec, coords)
                loss = F.mse_loss(x_pred, patches)
                val_loss += loss.item() * B

        if len(val_set) > 0:
            val_loss /= len(val_set)
        else:
            val_loss = float("nan")

        epoch_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train MSE: {train_loss:.4e} "
            f"- val MSE: {val_loss:.4e}"
        )

        logger.log(
            epoch=epoch,
            train_mse=train_loss,
            val_mse=val_loss,
        )
        if epoch % args.checkpoint_interval == 0:
            # save checkpoint every few epochs
            ckpt_path = os.path.join(args.output_dir, f"model_epoch{epoch}.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "num_classes": dataset.num_classes,
                    "idx_to_class": dataset.idx_to_class,
                    "patch_size": args.patch_size,
                    "T": args.T,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "schedule": args.schedule,
                },
                ckpt_path,
            )

            # save loss curve
            plot_loss_curve(
                epoch_losses,
                val_losses,
                os.path.join(args.output_dir, "loss_curve.png"),
            )

            # optionally visualize a random validation sample denoising
            if len(val_set) > 0:

                # get random number between 0 and len(val_set)
                random_val_idx = random.choice(val_set.indices)
                sample_rec = dataset.records[random_val_idx]

                sample_tomo, sample_particle = sample_rec["experiment"], sample_rec["particle_type"]

                patches, class_vec, coords = dataset[random_val_idx]
                patches = patches.to(device)
                class_vec = class_vec.to(device)
                coords = coords.to(device)
                patches = patches.unsqueeze(0).to(device)
                class_vec = class_vec.unsqueeze(0).to(device)
                coords = coords.unsqueeze(0).to(device)
                B = patches.size(0)
                t_vis = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)
                noise_vis = torch.randn_like(patches)
                x_noisy_vis = schedule.q_sample(patches, t_vis, noise=noise_vis)
                x_pred_vis = model(x_noisy_vis, t_vis, class_vec, coords)

                # take first element
                save_volume_slices(
                    patches[0, 0],
                    os.path.join(args.output_dir, f"epoch{epoch}_clean.png"),
                    title=f"Clean ({sample_tomo} - {sample_particle})",
                )
                save_volume_slices(
                    x_noisy_vis[0, 0],
                    os.path.join(args.output_dir, f"epoch{epoch}_noisy.png"),
                    title=f"Noisy ({sample_tomo} - {sample_particle})",
                )
                save_volume_slices(
                    x_pred_vis[0, 0],
                    os.path.join(args.output_dir, f"epoch{epoch}_denoised.png"),
                    title=f"Denoised Predicted ({sample_tomo} - {sample_particle})",
                )


if __name__ == "__main__":
    main()
