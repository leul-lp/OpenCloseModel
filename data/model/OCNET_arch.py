import torch.nn.functional as F
import torch.nn as nn

class OCNet(nn.Module):
    def __init__(self, input_shape=63, dropout_rate=0.2):
        super(OCNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 80)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(80, 50)


        self.dropout_3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout_1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout_2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout_3(x)


        x = self.fc4(x)

        return x        