import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
