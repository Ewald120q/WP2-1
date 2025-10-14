import torch
import torch.nn as nn
import torch.nn.functional as F


class DMTimeBinaryClassificator241002_1(nn.Module):
    def __init__(self, resol):
        super(DMTimeBinaryClassificator241002_1, self).__init__()
        self.resol = resol
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            # After conv: (256-5+1) = 252
            self.fc1 = nn.Linear(16 * 252 * 252, 256)
        elif resol == 128:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            # After conv: (128-5+1) = 124
            self.fc1 = nn.Linear(8 * 124 * 124, 128)
        elif resol == 64:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            # After conv: (64-5+1) = 60
            self.fc1 = nn.Linear(8 * 60 * 60, 256)
        elif resol == 32:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            # After conv: (32-5+1) = 28
            self.fc1 = nn.Linear(16 * 28 * 28, 256)
            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_2(nn.Module):
    def __init__(self, resol):
        super(DMTimeBinaryClassificator241002_2, self).__init__()
        self.resol = resol
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)
            # After conv1: (256-5+1) = 252, after pool: 126, after conv2: (126-5+1) = 122
            self.fc1 = nn.Linear(16 * 122 * 122, 512)
        elif resol == 128:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)
            # After conv1: (128-5+1) = 124, after pool: 62, after conv2: (62-5+1) = 58
            self.fc1 = nn.Linear(16 * 58 * 58, 128)
        elif resol == 64:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)
            # After conv1: (64-5+1) = 60, after pool: 30, after conv2: (30-5+1) = 26
            self.fc1 = nn.Linear(16 * 26 * 26, 128)
        elif resol == 32:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)
            # After conv1: (32-5+1) = 28, after pool: 14, after conv2: (14-5+1) = 10
            self.fc1 = nn.Linear(16 * 10 * 10, 512)
            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_3(nn.Module):
    def __init__(self, resol):
        super(DMTimeBinaryClassificator241002_3, self).__init__()
        self.resol = resol
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)
            # After conv1: 252, after pool1: 126, after conv2: 122, after pool2: 61, after conv3: 57
            self.fc1 = nn.Linear(12 * 57 * 57, 256)
        elif resol == 128:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 12, kernel_size=5, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(12, 12, kernel_size=5, padding=0)
            # After conv1: 124, after pool1: 62, after conv2: 58, after pool2: 29, after conv3: 25
            self.fc1 = nn.Linear(12 * 25 * 25, 256)
        elif resol == 64:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 12, kernel_size=5, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(12, 8, kernel_size=5, padding=0)
            # After conv1: 60, after pool1: 30, after conv2: 26, after pool2: 13, after conv3: 9
            self.fc1 = nn.Linear(8 * 9 * 9, 512)
        elif resol == 32:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 12, kernel_size=5, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(12, 12, kernel_size=5, padding=0)
            # After conv1: 28, after pool1: 14, after conv2: 10, after pool2: 5, after conv3: 1
            self.fc1 = nn.Linear(12 * 1 * 1, 512)
            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def model_DM_time_binary_classificator_241002_1(resol):
    return DMTimeBinaryClassificator241002_1(resol)

def model_DM_time_binary_classificator_241002_2(resol):
    return DMTimeBinaryClassificator241002_2(resol)

def model_DM_time_binary_classificator_241002_3(resol):
    return DMTimeBinaryClassificator241002_3(resol)

models_htable = {
    'DM_time_binary_classificator_241002_1': model_DM_time_binary_classificator_241002_1,
    'DM_time_binary_classificator_241002_2': model_DM_time_binary_classificator_241002_2,
    'DM_time_binary_classificator_241002_3': model_DM_time_binary_classificator_241002_3
}

