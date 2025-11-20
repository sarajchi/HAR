import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
from dataloader import HARDataSet
from fusion import FusionModel
from MoViNet.config import _C as config
import numpy as np
import torchvision.transforms as transforms
import seaborn as sns


def create_data_loaders(dataset, batch_size=16):
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    valid_size = int(0.15 * total_size)
    test_size = total_size - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    return train_loader, valid_loader, test_loader


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_labels = []
    all_scores = []  # Initialize list to store all softmax outputs

    with torch.no_grad():
        for video_frames, imu_data, actions in loader:
            video_frames, imu_data, actions = video_frames.to(device), imu_data.to(device), actions.to(device)
            outputs = model(video_frames, imu_data)
            loss = criterion(outputs, actions)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == actions).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(actions.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())  # Save softmax outputs

    accuracy = 100 * total_correct / len(loader.dataset)
    return total_loss / len(loader), accuracy, np.array(all_preds), np.array(all_labels), np.array(all_scores)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Fusion Model')
    plt.savefig('Confusion Matrix.pdf')
    plt.show()


def plot_precision_recall_curve(all_labels, all_scores, class_names):
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(all_labels == i, all_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name}')

    plt.title('Precision-Recall Curve for Fusion Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best', shadow=True)  # Use 'best' location for the legend
    plt.grid(True)
    plt.savefig('Precision-Recall Curve.pdf')
    plt.show()


def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, device, num_epochs=3):
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
        print(train_accuracy)
        if (epoch + 1) % 5 == 0:
            test_loss, test_accuracy, all_preds, all_labels, all_scores = evaluate_model(model, test_loader, criterion,
                                                                                         device)
            print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%')

    # Final evaluation on test data
    test_loss, test_accuracy, all_preds, all_labels, all_scores = evaluate_model(model, test_loader, criterion, device)

    print(f'Final Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%')
    print(classification_report(all_labels, all_preds, target_names=list(dataset.action_to_idx.keys())))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, list(dataset.action_to_idx.keys()))
    plot_precision_recall_curve(all_labels, all_scores, list(dataset.action_to_idx.keys()))


if __name__ == '__main__':
    # Configurations and model setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = HARDataSet(root_dir='./dataset', transform=transform)
    train_loader, valid_loader, test_loader = create_data_loaders(dataset)

    movinet_config = config.MODEL.MoViNetA0
    model = FusionModel(movinet_config, num_classes=3, lstm_input_size=12, lstm_hidden_size=512, lstm_num_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, device)
