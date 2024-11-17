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

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, state: Union[npt.NDArray[np.float_], Tensor]) -> Tensor:
        """
        Runs a forward pass of the neural network

        Inputs:
            state:      state input

        Outputs:
            mlp_out:    MLP output
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        layer1_output = F.relu(self.layer1(state))
        layer2_output = F.relu(self.layer2(layer1_output))
        mlp_out = self.layer3(layer2_output)

        return mlp_out
