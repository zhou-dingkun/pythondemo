import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 implementation using PyTorch with ReLU activation functions.
    Architecture: Input -> Conv1 -> Pool2 -> Conv3 -> Pool4 -> FC5 -> FC6 -> Output
    
    Original paper: "Gradient-Based Learning Applied to Document Recognition" by Yann LeCun et al.
    This implementation uses ReLU instead of tanh for better performance.
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize LeNet-5 model.
        
        Args:
            num_classes (int): Number of output classes (default: 10 for MNIST)
        """
        super(LeNet5, self).__init__()
        
        # Conv1: Input (1x32x32) -> (6x28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        # Pool2: (6x28x28) -> (6x14x14)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Conv3: (6x14x14) -> (16x10x10)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # Pool4: (16x10x10) -> (16x5x5)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # FC5: (16*5*5=400) -> 120
        self.fc5 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        
        # FC6: 120 -> 84
        self.fc6 = nn.Linear(in_features=120, out_features=84)
        
        # Output: 84 -> num_classes
        self.output = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 32, 32)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Conv1 -> ReLU -> Pool2
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv3 -> ReLU -> Pool4
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC5 -> ReLU
        x = self.fc5(x)
        x = F.relu(x)
        
        # FC6 -> ReLU
        x = self.fc6(x)
        x = F.relu(x)
        
        # Output
        x = self.output(x)
        
        return x
