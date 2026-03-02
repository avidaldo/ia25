import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # [1]
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# The input size here depends on the output size of the last conv/pool layer
# Original image 32x32
# After conv1 (kernel 5): the window is 5x5, so 2 pixels are reduced on each side
#  32-2-2 = 28 -> 28x28
# After pool1 (kernel 2, stride 2): Basically one pixel every 2x2, so it's reduced by half in each dimension
#  28/2 = 14 -> 14x14
# After conv2 (kernel 5): 14-(5-1) = 10 -> 10x10
# pool(conv2) -> 10/2 = 5 -> 5x5
# flattening: 16 channels * 5 * 5 