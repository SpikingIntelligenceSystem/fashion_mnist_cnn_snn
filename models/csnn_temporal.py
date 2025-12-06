import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils


class FashionCSNN_Temporal(nn.Module):
    def __init__(self, beta=0.9, num_classes=10, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.atan()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
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

    def forward(self, spk_in):
        utils.reset(self)  # Reset hidden state
        T, batch, _ = spk_in.shape
        mem_rec = []

        for t in range(T):
            x_t = spk_in[t].view(batch, 1, 28, 28)  # Reformat image shape
            x_t = self.conv1(x_t)
            x_t = self.lif1(x_t)
            x_t = self.conv2(x_t)
            x_t = self.lif2(x_t)
            x_t = self.pool(x_t)
            x_t = x_t.view(x_t.size(0), -1)  # Flatten
            x_t = self.fc1(x_t)
            x_t = self.lif_conn(x_t)
            x_t = self.fc2(x_t)
            _, mem_out = self.lif_out(x_t)
            mem_rec.append(mem_out)
        mem_rec = torch.stack(mem_rec)
        logits = mem_rec.mean(dim=0)
        return logits, mem_rec
    # Pass data through network and return logits and membrane
