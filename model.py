import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.01),

            nn.MaxPool2d(2,2),

            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),

            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),

            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 20, 3),
            nn.ReLU(),
            nn.BatchNorm2d(20),

            nn.Conv2d(20, 10, 3),

            nn.AvgPool2d(3, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x 