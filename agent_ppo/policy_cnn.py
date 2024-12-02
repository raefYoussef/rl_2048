import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from typing import Union


class PolicyCNN(nn.Module):
    """
    Convolutional Neural Network for policy representation.
    Extracts spatial features using CNN layers, followed by fully connected layers.
    """

    def __init__(
        self, in_dim: tuple[int, int], out_dim: int, hidden_dim: int = 64
    ) -> None:
        """
        Initialize the network with convolutional and fully connected layers.

        Inputs:
            in_dim:     Input dimensions (height, width) as a tuple
            out_dim:    Output dimensions (e.g., number of actions)
            hidden_dim: Number of neurons in the fully connected layer
        """
        super(PolicyCNN, self).__init__()

        if len(in_dim) == 2:  # log2 grid
            num_chan = 1
        else:
            num_chan = in_dim[0]

        # Define convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=num_chan, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        # Compute the flattened size after convolutions
        conv_out_dim = self._get_conv_output(in_dim)

        # Define fully connected layers
        self.fc1 = nn.Linear(conv_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def _get_conv_output(
        self, shape: Union[tuple[int, int, int], tuple[int, int]]
    ) -> int:
        """
        Helper function to calculate the output size of the CNN layers.

        Inputs:
            shape: Input shape (height, width)

        Outputs:
            Flattened size after convolutions
        """
        if len(shape) == 2:  # log2 grid
            dummy_input = torch.zeros(1, 1, *shape)  # Batch size 1, single channel
        else:
            dummy_input = torch.zeros(1, *shape)  # Batch size 1
        output = self.conv2(self.conv1(dummy_input))
        return int(np.prod(output.size()))

    def forward(self, state: Union[npt.NDArray[np.float64], Tensor]) -> Tensor:
        """
        Runs a forward pass of the neural network.

        Inputs:
            state:      State input (B,C,H,W)

        Outputs:
            cnn_out:    Output of the policy network
        """

        # Convolutional layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))

        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        cnn_out = self.fc2(x)

        return cnn_out
