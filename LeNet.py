import torch
import torch.nn as nn

class LeNet1(nn.Module):
    
    def __init__(self):
        super(LeNet1, self).__init__()
        
        # input is Nx1x28x28
        model_list = [
            # params: 4*(5*5*1 + 1) = 104
            # output is (28 - 5) + 1 = 24 => Nx4x24x24
            nn.Conv2d(1, 4, 5),
            nn.Tanh(),
            # output is 24/2 = 12 => Nx4x12x12
            nn.AvgPool2d(2),
            # params: (5*5*4 + 1) * 12 = 1212
            # output: 12 - 5 + 1 = 8 => Nx12x8x8
            nn.Conv2d(4, 12, 5),
            nn.Tanh(),
            # output: 8/2 = 4 => Nx12x4x4
            nn.AvgPool2d(2)
        ]
        
        self.model = nn.Sequential(*model_list)
        # params: (12*4*4 + 1) * 10 = 1930
        self.fc = nn.Linear(12*4*4, 10)
        self.criterion = nn.CrossEntropyLoss()
        
        # Total number of parameters = 104 + 1212 + 1930 = 3246
    
    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out, self.model(x)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        # feature_map = y.clone()
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        feature_map = y.clone()
        y = self.pool2(y)
        # feature_map = y.clone()
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        # feature_map = y.clone()
        y = self.fc2(y)
        y = self.relu4(y)
        
        y = self.fc3(y)

        return y, feature_map