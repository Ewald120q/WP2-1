import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import ResNet, BasicBlock, Bottleneck
from training_models_base import BinaryClassifierBase

from training_models_GAP import *


class LateFusionCombinedDMFTModel(nn.Module):
    def __init__(self, device, model_dmt, model_ft, k, freeze_towers=True):
        super().__init__()
        self.device = device

        self.model_dmt = model_dmt
        self.model_ft = model_ft
        self.k = k

        self.fc_dmt = nn.Linear(self.model_dmt.out_features, k)
        self.fc_ft  = nn.Linear(self.model_ft.out_features,  k)
        self.fc_out = nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

        if freeze_towers:
            for p in self.model_dmt.parameters():
                p.requires_grad = False
            for p in self.model_ft.parameters():
                p.requires_grad = False
                
            # Batch-Norm und Dropout stabil halten
            self.model_dmt.eval()
            self.model_ft.eval()

    def forward(self, batch):
        x_dmt = batch["dm_time"].to(self.device).unsqueeze(1)
        x_ft  = batch["freq_time"].to(self.device).unsqueeze(1)

        z_dmt = self.model_dmt.classifier_features(x_dmt)         
        z_ft  = self.model_ft.classifier_features(x_ft)           

        z_dmt = self.fc_dmt(z_dmt)                                
        z_ft  = self.fc_ft(z_ft)                                  

        z = z_dmt * z_ft                                          
        return self.fc_out(z)
    
    def classifier(self, x):
        raise NameError("classifier not implemented for LateFusionCombinedDMFTModel. Why do u need that?")
        
class MidFusionCombinedDMFTModel(nn.Module):
    def __init__(self, device, model_dmt, model_ft, k, freeze_towers=True):
        super().__init__()
        self.device = device

        self.model_dmt = model_dmt
        self.model_ft = model_ft
        self.k = k

        self.fc_dmt = nn.Linear(self.model_dmt.out_features, k)
        self.fc_ft  = nn.Linear(self.model_ft.out_features,  k)
        self.fc_head = self.backbone = ResNet(
                                block=BasicBlock,
                                layers=[2, 2, 2, 2],
                                num_classes=2,
                                in_channels=128,
                            )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )

        if freeze_towers:
            for p in self.model_dmt.parameters():
                p.requires_grad = False
            for p in self.model_ft.parameters():
                p.requires_grad = False
                
            # Batch-Norm und Dropout stabil halten
            self.model_dmt.eval()
            self.model_ft.eval()

    def forward(self, batch):
        x_dmt = batch["dm_time"].to(self.device).unsqueeze(1)  
        x_ft  = batch["freq_time"].to(self.device).unsqueeze(1)

        z_dmt = self.model_dmt.classifier_mid_level_features(x_dmt)         
        z_ft  = self.model_ft.classifier_mid_level_features(x_ft)
        
        z = torch.cat((z_dmt, z_ft), dim=1)
        z = self.fuse(z)
        
        return self.fc_head(z)
    
    def classifier(self, x):
        raise NameError("classifier not implemented for LateFusionCombinedDMFTModel. Why do u need that?")
        


class DMTimeBinaryClassificator241002_1(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_1 aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # conv1: (B, C_in, 256, 256) -> (B, 16, 252, 252)
        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=5, padding=0)
        # After conv: (256-5+1) = 252
        self.fc1 = nn.Linear(16 * 252 * 252, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
        
        self.out_features = self.fc1.out_features
    
    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return x
    
    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.fc2(x)
        return x

class DMTimeBinaryClassificator241002_2(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_2 aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # conv1: (256 -> 252), pool1: (252 -> 126), conv2: (126 -> 122)
        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)
        # After conv1: (256-5+1) = 252, after pool: 126, after conv2: (126-5+1) = 122
        self.fc1 = nn.Linear(16 * 122 * 122, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
        
        self.out_features = self.fc1.out_features
        
    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_3(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_3 aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # After conv1: 252, after pool1: 126, after conv2: 122, after pool2: 61, after conv3: 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)

        self.fc1 = nn.Linear(12 * 57 * 57, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
        
        self.out_features = self.fc1.out_features
        
        
    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_4(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_4 aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # conv1: 256 -> 252, pool1: 252 -> 126
        # conv2: 126 -> 122, conv2b: 122 -> 122, pool2: 122 -> 61
        # conv3: 61 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)  # 252
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 126
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)  # 122
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 122
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 61
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)  # 57

        self.fc1 = nn.Linear(12 * 57 * 57, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
        
        self.out_features = self.fc1.out_features

    def classifier_features(self, x):
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
        return x
    
    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_5(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_5 aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # conv1: 256 -> 252, conv1b: 252 -> 252, pool1: 252 -> 126
        # conv2: 126 -> 122, conv2b: 122 -> 122, pool2: 122 -> 61
        # conv3: 61 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)  # 252
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 252
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 126
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)  # 122
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 122
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 61
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)  # 57

        self.fc1 = nn.Linear(12 * 57 * 57, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
        
        self.out_features = self.fc1.out_features
        
    def classifier_features(self, x):
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
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_6(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_6 aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # conv1: 256 -> 252, conv1b: 252 -> 252, pool1: 252 -> 126
        # conv2: 126 -> 122, conv2b: 122 -> 122, pool2: 122 -> 61
        # conv3: 61 -> 57, conv3b: 57 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)  # 252
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 252
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 126
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)  # 122
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 122
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 61
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)  # 57
        self.conv3b = nn.Conv2d(12, 12, kernel_size=3, padding=1)  # 57

        self.fc1 = nn.Linear(12 * 57 * 57, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 2)
        
        self.out_features = self.fc1.out_features
        
    def classifier_features(self, x):
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
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.fc2(x)
        return x
    
class DMTimeBinaryClassificator241002_6_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_6_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # conv1: 256 -> 252, conv1b: 252 -> 252, pool1: 252 -> 126
        # conv2: 126 -> 122, conv2b: 122 -> 122, pool2: 122 -> 61
        # conv3: 61 -> 57, conv3b: 57 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)  # 252
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 252
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 126
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)  # 122
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 122
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 61
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)  # 57
        self.conv3b = nn.Conv2d(12, 12, kernel_size=3, padding=1)  # 57

        #self.fc1 = nn.Linear(12 * 57 * 57, 256)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(12, 2)
        
        self.out_features = 12*47*57 #self.fc1.out_features
        
    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = self.dropout_conv(x)
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = F.relu(self.gap(x))
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

class DMTimeBinaryClassificator241002_6_v2(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_6_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # conv1: 256 -> 252, conv1b: 252 -> 252, pool1: 252 -> 126
        # conv2: 126 -> 122, conv2b: 122 -> 122, pool2: 122 -> 61
        # conv3: 61 -> 57, conv3b: 57 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=5, padding=0)  # 252
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 252
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 126
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, padding=0)  # 122
        self.conv2b = nn.Conv2d(16, 8, kernel_size=3, padding=1)  # 122
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 61
        self.conv3 = nn.Conv2d(8, 8, kernel_size=5, padding=0)  # 57
        self.conv3b = nn.Conv2d(8, 4, kernel_size=3, padding=1)  # 57

        self.fc1 = nn.Linear(4 * 57 * 57, 256)
        #self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(256, 2)
        
        self.out_features = self.fc1.out_features
        
    def classifier_features(self, x):
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
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
    


class DMTimeBinaryClassificator241002_6_v3(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(f"Optimiert für resol=256, bekommen: {resol}")

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=5)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4) # 57 -> 14
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 14 -> 7

        self.fc1 = nn.Linear(64 * 7 * 7, 256) 
        self.fc2 = nn.Linear(256, 2)
        
        self.out_features = self.fc1.out_features

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.pool4(x)
        
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x




class DMTimeBinaryClassificatorResNet18(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"ResNet18-Wrapper aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # ResNet18: BasicBlock + [2,2,2,2]
        # in_channels hängt vom mode ab (1 für dmt/ft, 2 für dmft)
        self.backbone = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=2,
            in_channels=self.input_channels,
            dropout_prob=dropout,
        )
        
        self.out_features = self.backbone.fc.in_features

    def classifier(self, x):
        return self.backbone(x)
    
    def classifier_features(self, x):
        return self.backbone.forward_features(x)
    
    def classifier_mid_level_features(self, x):
        return self.backbone.forward_mid_level_features(x)


class DMTimeBinaryClassificatorResNet50(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"ResNet50-Dropout-Wrapper aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # ResNet50: Bottleneck + [3,4,6,3]
        self.backbone = ResNet(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=2,
            in_channels=self.input_channels,
            dropout_prob=dropout,
        )
        
        self.out_features = 512 * self.backbone.fc.in_features

    def classifier(self, x):
        return self.backbone(x)

    def classifier_features(self, x):
        return self.backbone.forward_features(x)


def model_DM_time_binary_classificator_241002_1(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_1(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_2(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_2(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_3(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_3(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_4(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_4(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_5(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_5(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_6(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_6(resol, mode, dropout, device)

def model_DM_time_binary_classificator_241002_2_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_2_GAP(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_3_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_3_GAP(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_4_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_4_GAP(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_5_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_5_GAP(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_6_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_6_GAP(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_7_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_7_GAP(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_8_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_8_GAP(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_9_GAP(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_9_GAP(resol, mode, dropout, device)

def model_DM_time_binary_classificator_241002_6_v2(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_6_v2(resol, mode, dropout, device)
def model_DM_time_binary_classificator_241002_6_v3(resol, mode, dropout, device):
    return DMTimeBinaryClassificator241002_6_v3(resol, mode, dropout, device)
def model_DM_time_binary_classificator_resnet18(resol, mode, dropout, device):
    return DMTimeBinaryClassificatorResNet18(resol, mode, dropout, device)
def model_DM_time_binary_classificator_resnet50(resol, mode, dropout, device):
    return DMTimeBinaryClassificatorResNet50(resol, mode, dropout, device)
def model_LateFusionCombinedDMFTModel(resol, mode, dropout, device, model_dmt, model_ft, k, freeze_towers):
    return LateFusionCombinedDMFTModel(resol, mode, dropout, device, model_dmt, model_ft, k, freeze_towers)
def model_MidFusionCombinedDMFTModel(resol, mode, dropout, device, model_dmt, model_ft, k, freeze_towers):
    return MidFusionCombinedDMFTModel(device, model_dmt, model_ft, k, freeze_towers)


models_htable = {
    'DM_time_binary_classificator_241002_1': model_DM_time_binary_classificator_241002_1,
    'DM_time_binary_classificator_241002_2': model_DM_time_binary_classificator_241002_2,
    'DM_time_binary_classificator_241002_3': model_DM_time_binary_classificator_241002_3,
    'DM_time_binary_classificator_241002_4': model_DM_time_binary_classificator_241002_4,
    'DM_time_binary_classificator_241002_5': model_DM_time_binary_classificator_241002_5,
    'DM_time_binary_classificator_241002_6': model_DM_time_binary_classificator_241002_6,
    
    'DM_time_binary_classificator_241002_2_GAP': model_DM_time_binary_classificator_241002_2_GAP,
    'DM_time_binary_classificator_241002_3_GAP': model_DM_time_binary_classificator_241002_3_GAP,
    'DM_time_binary_classificator_241002_4_GAP': model_DM_time_binary_classificator_241002_4_GAP,
    'DM_time_binary_classificator_241002_5_GAP': model_DM_time_binary_classificator_241002_5_GAP,
    'DM_time_binary_classificator_241002_6_GAP': model_DM_time_binary_classificator_241002_6_GAP,
    'DM_time_binary_classificator_241002_7_GAP': model_DM_time_binary_classificator_241002_7_GAP,
    'DM_time_binary_classificator_241002_8_GAP': model_DM_time_binary_classificator_241002_8_GAP,
    'DM_time_binary_classificator_241002_9_GAP': model_DM_time_binary_classificator_241002_9_GAP,
    
    'DM_time_binary_classificator_241002_6_v2': model_DM_time_binary_classificator_241002_6_v2,
    'DM_time_binary_classificator_241002_6_v3': model_DM_time_binary_classificator_241002_6_v3,

    'DM_time_binary_classificator_resnet18': model_DM_time_binary_classificator_resnet18,
    'DM_time_binary_classificator_resnet50': model_DM_time_binary_classificator_resnet50,
    'LateFusionCombinedDMFTModel' : model_LateFusionCombinedDMFTModel,
    'MidFusionCombinedDMFTModel' : model_MidFusionCombinedDMFTModel,
}
