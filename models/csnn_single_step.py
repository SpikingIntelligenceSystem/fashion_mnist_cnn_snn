import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils


class FashionCSNN_SingleStep(nn.Module):
    def __init__(self, beta=0.9, num_classes=10, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.atan()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1),
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.pool = nn.MaxPool2d(2, 2)
        # Define conv. layers

        self.fc1 = nn.Linear(64*14*14, 128)
        self.lif_conn = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(128, num_classes)
        self.lif_out = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        # Define connected layers

    def forward(self, x):
        utils.reset(self)   # Reset membrane hidden state
        x = self.conv1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.lif_conn(x)
        x = self.fc2(x)
        _, mem_out = self.lif_out(x)  # Spk not needed in this model
        return mem_out
    # Pass data through network and return membrane
