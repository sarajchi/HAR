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
    Dataset for HAR inference on a single Sample_* folder.

    Behaviour:
        - If called as HAR_Inference_DataSet(root_dir, transform):
            Uses the second last Sample_* folder in root_dir
            (original online behaviour; last one may not be fully written).

        - If called as HAR_Inference_DataSet(root_dir, transform, target_sample="Sample_37"):
            Uses exactly that Sample_37 folder (for offline processing / replay).
    """

    def __init__(self, root_dir, transform=None, target_sample=None):
        self.transform = transform
        self.action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}  # Action to index mapping

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

        # Decide which sample to use
        if target_sample is None:
            # Original behaviour: use second last if possible, otherwise last
            if len(samples) >= 2:
                chosen_sample = samples[-2]
            else:
                chosen_sample = samples[-1]
        else:
            if target_sample not in samples:
                raise ValueError(
                    f"Requested sample '{target_sample}' not found in {root_dir}"
                )
            chosen_sample = target_sample

        sample_path = root_dir / chosen_sample

        # Collect all frame paths (.jpg) in this sample
        images_path = [
            sample_path / f
            for f in os.listdir(sample_path)
            if f.lower().endswith('.jpg')
        ]
        images_path.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))

        imu_path = sample_path / 'imu.csv'

        if not images_path:
            raise RuntimeError(f"No .jpg frames found in {sample_path}")
        if not imu_path.exists():
            raise FileNotFoundError(f"IMU file not found: {imu_path}")

        # Store as a single sample: (list_of_frame_paths, imu_csv_path)
        self.Sample = [(images_path, imu_path)]
        self.SampleNumber = chosen_sample  # e.g. "Sample_37"

    def __len__(self):
        return len(self.Sample)

    def __getitem__(self, idx):
        frames_path, imu_path = self.Sample[idx]

        # Load frames
        frames = [Image.open(frame) for frame in frames_path]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Read IMU data
        imu_data = pd.read_csv(imu_path)

        # Frames: [sequence_length, C, H, W] -> [C, sequence_length, H, W]
        frames_tensor = torch.stack(frames)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)

        return frames_tensor, torch.tensor(imu_data.values, dtype=torch.float32)
