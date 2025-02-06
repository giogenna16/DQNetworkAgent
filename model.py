import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class FromPixelsQNetwork(nn.Module):
    """Q-Network for processing raw pixel observations."""

    def __init__(self, action_size, seed, fc1_units=512, fc2_units=512):
        """Initialize the CNN-based Q-Network.
        
        Params:
        ======
            action_size (int): Number of actions
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FromPixelsQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Convolutional layers
        # in_channels: The number of input channels (e.g., 3 for RGB images).
        # out_channels: The number of filters (feature maps) the layer learns.
        # kernel_size: The size of each filter (8x8 pixels).
        # stride: The step size the filter moves while scanning the image.
        # output: ((inputSize - kernel_size)/stride) + 1 
        # start = (1, 84, 84, 3), output = (84-8)/4 + 1 = 20
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)  # Output: (20, 20, 32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # Output: (9, 9, 64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # Output: (7, 7, 64)

        # Fully connected layers
        # start = (7, 7, 64)
        self.fc1 = nn.Linear(7 * 7 * 64, fc1_units)  # Flattened CNN output to FC layer
        self.fc2 = nn.Linear(fc2_units, action_size)  # Output layer (Q-values for each action)

    def forward(self, state):
        """Build a network that maps state (image) -> action values."""
        state = state.squeeze(1)
        x = state.permute(0, 3, 1, 2)  # Change from (batch, 84, 84, 3) to (batch, 3, 84, 84) for PyTorch
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the CNN output
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Output Q-values for each action

    
class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # shared fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Value stream
        self.value_fc = nn.Linear(fc2_units, 1)  # Output a single value V(s)
        # Advantage stream
        self.advantage_fc = nn.Linear(fc2_units, action_size)  # Output A(s, a)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value_fc(x)
        advantage = self.advantage_fc(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
    

class FromPixelsDuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed, fc1_units=512, fc2_units=512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FromPixelsDuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Convolutional layers
        # in_channels: The number of input channels (e.g., 3 for RGB images).
        # out_channels: The number of filters (feature maps) the layer learns.
        # kernel_size: The size of each filter (8x8 pixels).
        # stride: The step size the filter moves while scanning the image.
        # output: ((inputSize - kernel_size)/stride) + 1 
        # start = (1, 84, 84, 3), output = (84-8)/4 + 1 = 20
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)  # Output: (20, 20, 32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # Output: (9, 9, 64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # Output: (7, 7, 64)

        # Fully connected layers
        # start = (7, 7, 64)
        self.fc1 = nn.Linear(7 * 7 * 64, fc1_units)  # Flattened CNN output to FC layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)  # Output layer (Q-values for each action)

        # Value stream
        self.value_fc = nn.Linear(fc2_units, 1)  # Output a single value V(s)
        # Advantage stream
        self.advantage_fc = nn.Linear(fc2_units, action_size)  # Output A(s, a)


    def forward(self, state):
        """Build a network that maps state (image) -> action values (Q-values)."""
        state = state.squeeze(1)
        x = state.permute(0, 3, 1, 2)  # Change from (batch, 84, 84, 3) to (batch, 3, 84, 84) for PyTorch
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the CNN output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_fc(x)
        advantage = self.advantage_fc(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
