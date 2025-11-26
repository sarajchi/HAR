if __name__ == "__main__":
    print("\033cStarting ...\n")  # Clear Terminal

import os
import re
import sys
from pathlib import Path

try:
    import torch
    import pandas as pd
    from PIL import Image
    from torch.utils.data import Dataset
except ModuleNotFoundError as Err:
    missing_module = str(Err).replace('No module named ', '')
    missing_module = missing_module.replace("'", '')
    if missing_module == 'PIL':
        sys.exit(f'No module named {missing_module} try : pip install pillow')
    else:
        sys.exit(f'No module named {missing_module} try : pip install {missing_module}')


class HAR_Inference_DataSet(Dataset):
    """
    Dataset for ONLINE HAR inference using the *latest* Sample_* folders.

    Behaviour (online):
        - Called as HAR_Inference_DataSet(root_dir, transform)
          or HAR_Inference_DataSet(root_dir, transform, num_samples=2)

        - It:
            * Lists all Sample_* directories under `root_dir`
            * Sorts them numerically (Sample_1, Sample_2, ..., Sample_N)
            * Takes the last `num_samples` (by default, 2)
            * Concatenates:
                - all frames (.jpg) from these folders into one temporal sequence
                - all IMU rows (from imu.csv) into one time-series (row-wise)

        - The dataset always contains exactly ONE item:
              (frames_tensor, imu_tensor)

          where:
              frames_tensor shape: [C, T, H, W]
              imu_tensor    shape: [T, 12]

        - The list of underlying sample names is stored in:
              self.sample_names  (e.g. ["Sample_37", "Sample_38"])
    """

    def __init__(self, root_dir, transform=None, num_samples: int = 2):
        self.transform = transform
        self.action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}  # kept for consistency

        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

        # List all Sample_* directories
        samples = [
            d.name
            for d in root_dir.iterdir()
            if d.is_dir() and d.name.startswith("Sample_")
        ]
        if not samples:
            raise IndexError(f"No Sample_* folders found in {root_dir}")

        # Sort numerically: Sample_1, Sample_2, ..., Sample_N
        samples.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        # Take the last `num_samples` samples (or all if fewer exist)
        num_samples = max(1, int(num_samples))
        recent_samples = samples[-num_samples:]

        self.sample_names = recent_samples  # e.g. ["Sample_37", "Sample_38"]

        # ----- Collect all frame paths and IMU csv paths -----
        all_frame_paths: list[Path] = []
        imu_paths: list[Path] = []

        for sname in recent_samples:
            sample_path = root_dir / sname

            # Frames
            images_path = [
                sample_path / f
                for f in os.listdir(sample_path)
                if f.lower().endswith('.jpg')
            ]
            images_path.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))
            all_frame_paths.extend(images_path)

            # IMU
            imu_path = sample_path / 'imu.csv'
            if not imu_path.exists():
                raise FileNotFoundError(f"IMU file not found: {imu_path}")
            imu_paths.append(imu_path)

        if not all_frame_paths:
            raise RuntimeError(
                f"No .jpg frames found in the last {len(recent_samples)} sample(s) "
                f"under {root_dir}"
            )

        # Store as a single logical sample: (list_of_frame_paths, list_of_imu_paths)
        self.Sample = [(all_frame_paths, imu_paths)]

    def __len__(self):
        # Always one combined sample (last N Sample_* folders concatenated)
        return len(self.Sample)

    def __getitem__(self, idx):
        frame_paths, imu_paths = self.Sample[idx]

        # ---- Load frames ----
        frames = [Image.open(frame) for frame in frame_paths]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # ---- Load and concatenate IMU data from all selected samples ----
        imu_df_list = [pd.read_csv(p) for p in imu_paths]

        # Concatenate row-wise (time axis), keep feature dimension the same
        imu_data = pd.concat(imu_df_list, axis=0, ignore_index=True)

        # Safety: if more than 12 columns somehow appear, keep only the first 12
        if imu_data.shape[1] > 12:
            imu_data = imu_data.iloc[:, :12]

        # Frames: [sequence_length, C, H, W] -> [C, sequence_length, H, W]
        frames_tensor = torch.stack(frames)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)

        imu_tensor = torch.tensor(imu_data.values, dtype=torch.float32)

        return frames_tensor, imu_tensor
