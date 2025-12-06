import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch.spikegen as spikegen
from models.csnn_temporal import FashionCSNN_Temporal
import os
import json
import time

"""
To Initiate Training, Paste Into Terminal In Project Root:
    py -m train_scripts.train_csnn_temporal
"""

# Model configurations
batch_size = 128
num_epochs = 10
num_steps = 50
learn_rate = 1e-4
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
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)
    return train_loader, test_loader


def spike_encode(images, num_steps, device):
    images = images.to(device)
    batch = images.size(0)
    flatten = images.view(batch, -1)
    spikes = spikegen.rate(flatten, num_steps)
    return spikes
# Rate encodes images into spikes


def train_model(model, loader, device, optimizer, criterion, num_steps, log_interval=5):
    model.train()
    current_loss = 0.0
    correct_ids = 0
    total_ids = 0
    num_batches = len(loader)
    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        labels = labels.to(device)
        spikes = spike_encode(images, num_steps, device)
        logits, _ = model(spikes)  # Import logits
        optimizer.zero_grad()

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        current_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct_ids += (preds == labels).sum().item()
        total_ids += labels.size(0)
        # Computation progress (if loop):
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            elapsed = time.time() - start_time
            print(f"Batch: {batch_idx}/{num_batches}")
            print(
                f"Loss: {loss.item():.3f} /// Elapsed Time: {elapsed:.2f} seconds.")
    epoch_loss = current_loss / total_ids
    epoch_accuracy = correct_ids / total_ids
    return epoch_loss, epoch_accuracy


def evaluate_model(model, loader, device, num_steps):
    model.eval()
    correct_ids = 0
    total_ids = 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            spikes = spike_encode(images, num_steps, device)

            logits, _ = model(spikes)
            preds = logits.argmax(dim=1)
            correct_ids += (preds == labels).sum().item()
            total_ids += labels.size(0)
    return correct_ids / total_ids  # Final accuracy


def main():
    set_seed(seed)
    device = identify_device()
    print(f"Training using: {device}")

    train_loader, test_loader = establish_dataloaders(batch_size)

    model = FashionCSNN_Temporal().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    epoch_losses = []
    epoch_accs = []

    for epoch in range(num_epochs):
        loss, accuracy = train_model(
            model, train_loader, device, optimizer, criterion, num_steps)
        epoch_losses.append(loss)
        epoch_accs.append(accuracy)

        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(
            f"Training Loss: {loss:.4f} /// Training Accuracy: {accuracy*100:.2f}%")
    total_accuracy = evaluate_model(model, test_loader, device, num_steps)
    print(
        f"Final Evaluated Accuracy Of CSNN Single: {total_accuracy*100:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/csnn_single_loss.json", "w") as f:
        json.dump(epoch_losses, f)


if __name__ == "__main__":
    main()
