# data/particles_dataset.py

import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mrcfile


class CryoETParticlesDataset(Dataset):
    """
    Dataset for CryoET particle-centered subvolumes.

    Expects:
      - tomogram_root/ : directory with .mrc tomograms
      - csv_path : CSV with at least columns:
          ['experiment', 'particle_type', 'x', 'y', 'z']
        'experiment' should match the tomogram filename *without* extension.
        (e.g. experiment='TS_5_4.bin0.denoised' -> file 'TS_5_4.bin0.denoised.mrc')

    Each item returns:
      patch : (1, D, H, W) float32 tensor (normalized per-patch)
      class_onehot : (num_classes,) float32 tensor
      coords_norm : (3,) float32 tensor, coordinates normalized to [0,1]
    """

    def __init__(
        self,
        tomogram_root: str,
        csv_path: str,
        patch_size: int = 64,
        split_experiments: Optional[List[str]] = None,
        cache_volumes: bool = True,
    ):
        """
        Args:
            tomogram_root: path to directory with .mrc tomograms.
            csv_path: path to particles_all_bin0.csv (or similar).
            patch_size: size of cubic patch (default 64).
            split_experiments: if not None, only keep entries whose
                               'experiment' is in this list.
            cache_volumes: if True, keep loaded tomograms in memory.
        """
        super().__init__()
        self.tomogram_root = tomogram_root
        self.csv_path = csv_path
        self.patch_size = int(patch_size)
        self.half = self.patch_size // 2
        self.cache_volumes = cache_volumes

        df = pd.read_csv(csv_path)

        required_cols = {"experiment", "particle_type", "x", "y", "z"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        if split_experiments is not None:
            df = df[df["experiment"].isin(split_experiments)].reset_index(drop=True)

        # Filter to rows whose tomogram file actually exists
        records = []
        for _, row in df.iterrows():
            experiment = str(row["experiment"])
            tomo_path = self._find_tomo_path(experiment)
            if not os.path.exists(tomo_path):
                # silently skip missing tomograms
                continue
            records.append(
                dict(
                    experiment=experiment,
                    particle_type=str(row["particle_type"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                )
            )

        if not records:
            raise RuntimeError(
                "No valid records found in CSV after filtering & file existence check."
            )

        self.records: List[Dict] = records

        # Build class mapping
        particle_types = sorted({r["particle_type"] for r in records})
        self.class_to_idx: Dict[str, int] = {
            cls: i for i, cls in enumerate(particle_types)
        }
        self.idx_to_class: Dict[int, str] = {
            i: cls for cls, i in self.class_to_idx.items()
        }
        self.num_classes = len(self.class_to_idx)

        # Volume cache
        self.volumes: Dict[str, np.ndarray] = {}

    def _find_tomo_path(self, experiment: str) -> str:
        """
        Resolve experiment name to a .mrc path.

        We try:
          - <root>/<experiment>.mrc
          - if experiment already endswith('.mrc'), just join root.
        """
        if experiment.endswith(".mrc"):
            fname = experiment
        else:
            fname = experiment + ".mrc"
        return os.path.join(self.tomogram_root, fname)

    def _load_volume(self, experiment: str) -> np.ndarray:
        if self.cache_volumes and experiment in self.volumes:
            return self.volumes[experiment]

        tomo_path = self._find_tomo_path(experiment)
        if not os.path.exists(tomo_path):
            raise FileNotFoundError(f"Tomogram {tomo_path} not found.")

        with mrcfile.open(tomo_path, mode="r") as mrc:
            vol = np.array(mrc.data, dtype=np.float32)  # (Z, Y, X)

        if self.cache_volumes:
            self.volumes[experiment] = vol
        return vol

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        experiment = rec["experiment"]
        cls_name = rec["particle_type"]
        x, y, z = rec["x"], rec["y"], rec["z"]

        vol = self._load_volume(experiment)  # (Z, Y, X)
        Z, Y, X = vol.shape

        # Center coordinates: CSV is in (x, y, z) voxel coords.
        cx = int(round(x))
        cy = int(round(y))
        cz = int(round(z))

        # Bounds-aware cropping; if near edges, pad with zeros.
        x_min = cx - self.half
        x_max = cx + self.half
        y_min = cy - self.half
        y_max = cy + self.half
        z_min = cz - self.half
        z_max = cz + self.half

        # Initialize patch with zeros
        patch_np = np.zeros((self.patch_size, self.patch_size, self.patch_size),
                            dtype=np.float32)

        # Compute overlap with actual volume
        src_x_min = max(x_min, 0)
        src_x_max = min(x_max, X)
        src_y_min = max(y_min, 0)
        src_y_max = min(y_max, Y)
        src_z_min = max(z_min, 0)
        src_z_max = min(z_max, Z)

        # Compute corresponding indices in patch
        dst_x_min = src_x_min - x_min
        dst_x_max = dst_x_min + (src_x_max - src_x_min)
        dst_y_min = src_y_min - y_min
        dst_y_max = dst_y_min + (src_y_max - src_y_min)
        dst_z_min = src_z_min - z_min
        dst_z_max = dst_z_min + (src_z_max - src_z_min)

        patch_np[
            dst_z_min:dst_z_max,
            dst_y_min:dst_y_max,
            dst_x_min:dst_x_max,
        ] = vol[src_z_min:src_z_max, src_y_min:src_y_max, src_x_min:src_x_max]

        # Normalize per-patch (z-score)
        patch = torch.from_numpy(patch_np)
        mean = patch.mean()
        std = patch.std()
        patch = (patch - mean) / (std + 1e-8)
        patch = patch.unsqueeze(0)  # (1, D, H, W)

        # Class one-hot
        class_idx = self.class_to_idx[cls_name]
        class_onehot = torch.nn.functional.one_hot(
            torch.tensor(class_idx, dtype=torch.long),
            num_classes=self.num_classes,
        ).float()

        # Normalized coordinates in [0,1]
        coords_norm = torch.tensor(
            [x / float(X), y / float(Y), z / float(Z)],
            dtype=torch.float32,
        )

        return patch, class_onehot, coords_norm

    def get_class_mapping(self) -> Dict[str, int]:
        """Return mapping from class name to index."""
        return dict(self.class_to_idx)
