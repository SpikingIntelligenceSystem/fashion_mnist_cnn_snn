import torch.nn as nn


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*14*14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        # Network definition

    def forward(self, x):
        x = self.features(x)
        flatten = x.view(x.size(0), -1)
        logits = self.classifier(flatten)
        return logits
    # Pass data through network and return logits
