import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import ResNet, BasicBlock


class DMTimeBinaryClassificator241002_1(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificator241002_1, self).__init__()
        self.resol = resol
        self.use_freq_time = use_freq_time
        self.device = device
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            # After conv: (256-5+1) = 252
            self.fc1 = nn.Linear(16 * 252 * 252, 256)

            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        if self.use_freq_time:
            x  = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)
        
        x = torch.unsqueeze(x, 1)
            
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_2(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificator241002_2, self).__init__()
        self.resol = resol
        self.use_freq_time = use_freq_time
        self.device = device
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)
            # After conv1: (256-5+1) = 252, after pool: 126, after conv2: (126-5+1) = 122
            self.fc1 = nn.Linear(16 * 122 * 122, 512)

            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        if self.use_freq_time:
            x  = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)
        
        x = torch.unsqueeze(x, 1)  

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_3(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificator241002_3, self).__init__()
        self.resol = resol
        self.use_freq_time = use_freq_time
        self.device = device
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)
            # After conv1: 252, after pool1: 126, after conv2: 122, after pool2: 61, after conv3: 57
            self.fc1 = nn.Linear(12 * 57 * 57, 256)

            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        if self.use_freq_time:
            x  = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)
            
        x = torch.unsqueeze(x, 1)
            
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class DMTimeBinaryClassificator241002_3_dropout(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificator241002_3_dropout, self).__init__()
        self.resol = resol
        self.dropout_conv = nn.Dropout2d(p=0.2)
        self.dropout_fc = nn.Dropout(p=0.4)
        
        self.device = device
        
        self.use_freq_time = use_freq_time
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)
            # After conv1: 252, after pool1: 126, after conv2: 122, after pool2: 61, after conv3: 57
            self.fc1 = nn.Linear(12 * 57 * 57, 256)
            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        if self.use_freq_time:
            x  = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)
            
        x = torch.unsqueeze(x, 1)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
    
class DMTimeBinaryClassificator241002_4_dropout(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificator241002_4_dropout, self).__init__()
        self.resol = resol
        self.dropout_conv = nn.Dropout2d(p=0.2)
        self.dropout_fc = nn.Dropout(p=0.4)
        
        self.device = device
        
        self.use_freq_time = use_freq_time
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0) # 252
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 126
            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0) # 122
            self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1) #122
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 61
            self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0) #57

            self.fc1 = nn.Linear(12 * 57 * 57, 256)
            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        if self.use_freq_time:
            x  = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)
            
        x = torch.unsqueeze(x, 1)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
    
class DMTimeBinaryClassificator241002_5_dropout(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificator241002_5_dropout, self).__init__()
        self.resol = resol
        self.dropout_conv = nn.Dropout2d(p=0.2)
        self.dropout_fc = nn.Dropout(p=0.4)
        
        self.device = device
        
        self.use_freq_time = use_freq_time
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0) # 252
            self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1) #252
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 126
            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0) # 122
            self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1) #122
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 61
            self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0) #57

            self.fc1 = nn.Linear(12 * 57 * 57, 256)
            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        if self.use_freq_time:
            x  = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)
            
        x = torch.unsqueeze(x, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
    
class DMTimeBinaryClassificator241002_6_dropout(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificator241002_6_dropout, self).__init__()
        self.resol = resol
        self.dropout_conv = nn.Dropout2d(p=0.2)
        self.dropout_fc = nn.Dropout(p=0.4)
        
        self.device = device
        
        self.use_freq_time = use_freq_time
        
        if resol == 256:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0) # 252
            self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1) #252
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 126
            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0) # 122
            self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1) #122
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 61
            self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0) #57
            self.conv3b = nn.Conv2d(12, 12, kernel_size=3, padding=1) #57

            self.fc1 = nn.Linear(12 * 57 * 57, 256)
            
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
    
    def forward(self, x):
        if self.use_freq_time:
            x  = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)
            
        x = torch.unsqueeze(x, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

class DMTimeBinaryClassificatorResNet18(nn.Module):
    def __init__(self, resol, use_freq_time, device):
        super(DMTimeBinaryClassificatorResNet18, self).__init__()
        self.resol = resol
        self.use_freq_time = use_freq_time
        self.device = device
        
        if resol != 256:
            raise ValueError(f"ResNet18-Wrapper aktuell nur für resol=256 implementiert, bekommen: {resol}")

        # ResNet18: BasicBlock + [2,2,2,2]
        self.backbone = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=2,
            in_channels=1,
        )

    def forward(self, x):
        if self.use_freq_time:
            x = x["freq_time"].to(self.device)
        else:
            x = x["dm_time"].to(self.device)

        # (B, H, W) -> (B, 1, H, W)
        x = torch.unsqueeze(x, 1)

        x = self.backbone(x)
        return x



def model_DM_time_binary_classificator_241002_1(resol, use_freq_time, device):
    return DMTimeBinaryClassificator241002_1(resol, use_freq_time,device)

def model_DM_time_binary_classificator_241002_2(resol, use_freq_time, device):
    return DMTimeBinaryClassificator241002_2(resol, use_freq_time, device)

def model_DM_time_binary_classificator_241002_3(resol, use_freq_time, device):
    return DMTimeBinaryClassificator241002_3(resol, use_freq_time, device)

def model_DM_time_binary_classificator_241002_3_dropout(resol, use_freq_time, device):
    return DMTimeBinaryClassificator241002_3_dropout(resol, use_freq_time, device)

def model_DM_time_binary_classificator_241002_4_dropout(resol, use_freq_time, device):
    return DMTimeBinaryClassificator241002_4_dropout(resol, use_freq_time, device)

def model_DM_time_binary_classificator_241002_5_dropout(resol, use_freq_time, device):
    return DMTimeBinaryClassificator241002_5_dropout(resol, use_freq_time, device)

def model_DM_time_binary_classificator_241002_6_dropout(resol, use_freq_time, device):
    return DMTimeBinaryClassificator241002_6_dropout(resol, use_freq_time, device)

def model_DM_time_binary_classificator_resnet18(resol, use_freq_time, device):
    return DMTimeBinaryClassificatorResNet18(resol, use_freq_time, device)


models_htable = {
    'DM_time_binary_classificator_241002_1': model_DM_time_binary_classificator_241002_1,
    'DM_time_binary_classificator_241002_2': model_DM_time_binary_classificator_241002_2,
    'DM_time_binary_classificator_241002_3': model_DM_time_binary_classificator_241002_3,
    'DM_time_binary_classificator_241002_3_dropout': model_DM_time_binary_classificator_241002_3_dropout,
    'DM_time_binary_classificator_241002_4_dropout': model_DM_time_binary_classificator_241002_4_dropout,
    'DM_time_binary_classificator_241002_5_dropout': model_DM_time_binary_classificator_241002_5_dropout,
    'DM_time_binary_classificator_241002_6_dropout': model_DM_time_binary_classificator_241002_6_dropout,
    'DM_time_binary_classificator_resnet18': model_DM_time_binary_classificator_resnet18,
}

