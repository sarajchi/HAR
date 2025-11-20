import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import re
from torch.utils.data import DataLoader
class HARDataSet(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=10):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = []
        self.action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}  # Define the mapping from actions to indices

        for action in os.listdir(root_dir):
            action_path = os.path.join(root_dir, action)
            for frame_folder in os.listdir(action_path):
                frame_folder_path = os.path.join(action_path, frame_folder)
                frames = [os.path.join(frame_folder_path, f) for f in os.listdir(frame_folder_path) if f.endswith('.jpg')]
                frames.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
                imu_path = os.path.join(frame_folder_path, 'imu.csv')

                if len(frames) >= sequence_length:
                    # Sliding window with a step size of 1
                    for i in range(0, len(frames) - sequence_length + 1, 1):
                        self.samples.append((frames[i:i + sequence_length], imu_path, action, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, imu_path, action, start_idx = self.samples[idx]
        frames = [Image.open(frame) for frame in frames_path]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Read IMU data and slice it according to the starting index
        imu_data = pd.read_csv(imu_path).drop(columns=['time_stamp'])
        imu_data = imu_data.iloc[start_idx:start_idx + len(frames_path)].reset_index(drop=True)

        action_idx = self.action_to_idx[action]  # Convert action name to integer index
        action_tensor = torch.tensor(action_idx, dtype=torch.long)  # Convert to tensor

        # Rearrange frame tensor dimensions to [C, num_frames, H, W]
        frames_tensor = torch.stack(frames)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # Change from [sequence_length, C, H, W] to [C, sequence_length, H, W]

        return frames_tensor, torch.tensor(imu_data.values, dtype=torch.float32), action_tensor




# Transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
