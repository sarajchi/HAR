import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from dataloader import HARDataSet
import torchvision.transforms as transforms
from fusion import FusionModel
from MoViNet.config import _C as config

def create_data_loaders(dataset, batch_size=4):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)
    test_size = total_size - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    return train_loader, valid_loader, test_loader

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for video_frames, imu_data, actions in train_loader:
            video_frames, imu_data, actions = video_frames.to(device), imu_data.to(device), actions.to(device)
            optimizer.zero_grad()
            outputs = model(video_frames, imu_data)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train_correct += (predicted == actions).sum().item()
            total_train_samples += actions.size(0)

        train_accuracy = 100 * total_train_correct / total_train_samples

        model.eval()
        total_valid_loss = 0
        total_valid_correct = 0
        total_valid_samples = 0

        with torch.no_grad():
            for video_frames, imu_data, actions in valid_loader:
                video_frames, imu_data, actions = video_frames.to(device), imu_data.to(device), actions.to(device)
                outputs = model(video_frames, imu_data)
                loss = criterion(outputs, actions)
                total_valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid_correct += (predicted == actions).sum().item()
                total_valid_samples += actions.size(0)

        valid_accuracy = 100 * total_valid_correct / total_valid_samples

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_train_loss/len(train_loader)}, Train Acc: {train_accuracy}%, Validation Loss: {total_valid_loss/len(valid_loader)}, Validation Acc: {valid_accuracy}%')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = HARDataSet(root_dir='./dataset_new', transform=transform)
train_loader, valid_loader, test_loader = create_data_loaders(dataset)

movinet_config = config.MODEL.MoViNetA0
model = FusionModel(movinet_config, num_classes=3, lstm_input_size=12, lstm_hidden_size=512, lstm_num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20)
