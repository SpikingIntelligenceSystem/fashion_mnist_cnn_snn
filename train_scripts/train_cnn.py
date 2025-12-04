import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn_baseline import FashionCNN
import json
import os

"""
To Initiate Training, Paste Into Terminal In Project Root:
    py -m train_scripts.train_cnn
"""

# Model configurations
batch_size = 128
num_epochs = 10
learn_rate = 1e-3
seed = 0
# Model configurations


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identify_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def establish_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, loader, device, optimizer, criterion):
    model.train()
    current_loss = 0.0
    correct_ids = 0
    total_ids = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        current_loss += loss.item() * images.size(0)  # Counts current loss for epoch
        preds = logits.argmax(dim=1)
        correct_ids += (preds == labels).sum().item()  # Counts correct images
        total_ids += labels.size(0)  # Counts total identified images
    epoch_loss = current_loss / total_ids
    epoch_accuracy = correct_ids / total_ids
    return epoch_loss, epoch_accuracy


def evaluate_model(model, loader, device):
    model.eval()
    correct_ids = 0
    total_ids = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)
            correct_ids += (preds == labels).sum().item()
            total_ids += labels.size(0)
    return correct_ids / total_ids  # Final accuracy


def main():
    set_seed(seed)
    device = identify_device()
    print(f"Training using: {device}")

    train_loader, test_loader = establish_dataloaders(batch_size)

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    epoch_losses = []
    epoch_accs = []

    for epoch in range(num_epochs):
        loss, accuracy = train_model(model, train_loader,
                                     device, optimizer, criterion)
        epoch_losses.append(loss)
        epoch_accs.append(accuracy)

        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(
            f"Training Loss: {loss:.4f} /// Training Accuracy: {accuracy*100:.2f}%")
    total_accuracy = evaluate_model(model, test_loader, device)
    print(f"Final Evaluated Accuracy Of CNN: {total_accuracy*100:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/cnn_loss.json", "w") as f:
        json.dump(epoch_losses, f)


if __name__ == "__main__":
    main()
