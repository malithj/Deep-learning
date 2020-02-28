import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 50)
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 20)
        self.fc4 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dp1(x)
        x = self.fc4(x)
        return x


net = Net()
