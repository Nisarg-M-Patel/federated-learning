import torch.nn as nn
import torch.nn.functional as F
import math


class SingleMLP(nn.Module):
    def __init__(self, hidden_dim=50, expand_ratio=1.0, width_multiplier=1.0):
        super(SingleMLP, self).__init__()
        self.input_dim = 784  # mnist has image of 28*28
        self.hidden_dim = hidden_dim
        assert (
            expand_ratio >= 1.0
        ), f"Expand ratio needs to be >=1 but found {expand_ratio}"
        self.expand_ratio = expand_ratio
        self.width_multiplier = width_multiplier

        self.fc1 = nn.Linear(self.input_dim, self.hidden_layer_input_channel)
        self.fc2 = nn.Linear(
            self.hidden_layer_input_channel, self.hidden_layer_output_channel
        )
        self.fc3 = nn.Linear(self.hidden_layer_output_channel, 10)

    @property
    def hidden_layer_input_channel(self):
        return math.ceil(self.hidden_dim * self.width_multiplier)

    @property
    def hidden_layer_output_channel(self):
        return math.ceil(
            self.hidden_dim * self.width_multiplier * self.expand_ratio
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
