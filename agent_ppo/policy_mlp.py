import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from typing import Union


class PolicyMLP(nn.Module):
    """
    Multi-Layer Perceptron network with a hidden layer of 64 neurons
    This MLP network can be used to represent the actor/critic networks
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64) -> None:
        """
        Initialize the network and set up the layers.

        Inputs:
            in_dim:     input dimensions as an int
            out_dim:    output dimensions as an int

        Outputs:
            None
        """
        super(PolicyMLP, self).__init__()

        self.layer1 = nn.Linear(np.prod(in_dim).item(), hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, state: Tensor) -> Tensor:
        """
        Runs a forward pass of the neural network

        Inputs:
            state:      state input (B,C,H,W)

        Outputs:
            mlp_out:    MLP output
        """

        # expects grids (log2 or onehot) so we need to flatten each grid
        flattened_state = state.reshape(state.shape[0], -1)
        layer1_output = F.relu(self.layer1(flattened_state))
        layer2_output = F.relu(self.layer2(layer1_output))
        mlp_out = self.layer3(layer2_output)

        return mlp_out
