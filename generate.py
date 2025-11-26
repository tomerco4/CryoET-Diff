# generate.py

import argparse
import os

import numpy as np
import torch

from models.diffusion_unet3d import DiffusionUNet3D
from diffusion.schedule import DiffusionSchedule
from utils.visualization import save_volume_slices
import mrcfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D CryoET particle volumes from trained DDPM"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--class_index", type=int, required=True,
                        help="Integer class index [0..C-1].")
    parser.add_argument("--coords", type=str, default="0.5,0.5,0.5",
                        help="Normalized coords 'x,y,z' in [0,1].")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    num_classes = ckpt.get("num_classes")
    idx_to_class = ckpt.get("idx_to_class")

    patch_size = ckpt.get("patch_size", 64)
    T = ckpt.get("T", 1000)
    beta_start = ckpt.get("beta_start", 1e-4)
    beta_end = ckpt.get("beta_end", 2e-2)
    schedule_type = ckpt.get("schedule", "linear")

    model = DiffusionUNet3D(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    schedule = DiffusionSchedule(
        T=T,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule_type,
        device=device,
    )

    coords_list = [float(c) for c in args.coords.split(",")]
    if len(coords_list) != 3:
        raise ValueError("coords must be 'x,y,z' normalized in [0,1].")
    coords_np = np.array(coords_list, dtype=np.float32)

    for i in range(args.num_samples):
        # start from Gaussian noise
        x_t = torch.randn(
            1, 1, patch_size, patch_size, patch_size, device=device
        )

        # fixed conditioning
        class_index = int(args.class_index)
        if not (0 <= class_index < num_classes):
            raise ValueError(f"class_index must be in [0,{num_classes-1}], available classes for this checkpoint are: {idx_to_class}")

        class_onehot = torch.nn.functional.one_hot(
            torch.tensor([class_index], device=device, dtype=torch.long),
            num_classes=num_classes,
        ).float()

        coords = torch.from_numpy(coords_np).to(device).view(1, 3)

        for t_step in reversed(range(T)):
            t = torch.full((1,), t_step, device=device, dtype=torch.long)
            with torch.no_grad():
                x0_pred = model(x_t, t, class_onehot, coords)

            mean, log_var = schedule.p_sample_x0(x_t, t, x0_pred)

            if t_step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_var) * noise
            else:
                x_t = mean

        x_sample = x_t.detach().cpu()[0, 0]  # (D,H,W)

        # save PNG slices
        png_path = os.path.join(args.output_dir, f"sample_{i}_class_{class_index}.png")
        save_volume_slices(x_sample, png_path, title=f"{idx_to_class[class_index]} - Sample {i}")

        # save as MRC for ChimeraX
        mrc_path = os.path.join(args.output_dir, f"sample_{i}_class_{class_index}.mrc")
        with mrcfile.new(mrc_path, overwrite=True) as mrc:
            mrc.set_data(x_sample.numpy().astype(np.float32))
            mrc.voxel_size = 1.0  # arbitrary; adjust if needed

        print(f"Saved sample {i} (class - {idx_to_class[class_index]})-> {png_path}, {mrc_path}")


if __name__ == "__main__":
    main()
