import torch
import torch.nn as nn
import torch.nn.functional as F
from training_models_base import BinaryClassifierBase

class DMTimeBinaryClassificator241002_2_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_2_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 126 -> 122 
        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(16, 2)

        self.feature_map_shape = (16, 122, 122)
        self.out_features = 16

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)
        return x

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_3_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_3_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 126 -> 122 -> 61 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)   
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)                    
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)                   

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(12, 2)

        self.feature_map_shape = (12, 57, 57)
        self.out_features = 12

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        return x

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_4_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_4_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 126 -> 122 -> 122 -> 61 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)   
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)                      
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)                 

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(12, 2)

        self.feature_map_shape = (12, 57, 57)
        self.out_features = 12

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        return x

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_5_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_5_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 252 -> 126 -> 122 -> 122 -> 61 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)   
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)                      
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)                  

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(12, 2)

        self.feature_map_shape = (12, 57, 57)
        self.out_features = 12

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        return x

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_6_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_6_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 252 -> 126 -> 122 -> 122 -> 61 -> 57 -> 57
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)   
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)                      
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)                     
        self.conv3b = nn.Conv2d(12, 12, kernel_size=3, padding=1)                   

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(12, 2)

        self.feature_map_shape = (12, 57, 57)
        self.out_features = 12

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

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x
    
class DMTimeBinaryClassificator241002_7_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_7_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 252 -> 126 -> 122 -> 122 -> 61 -> 57 -> 57 -> 53
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)   
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)                      
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)                     
        self.conv3b = nn.Conv2d(12, 12, kernel_size=3, padding=1)                   
        self.conv4 = nn.Conv2d(12, 16, kernel_size=5, padding=0)                    

        self.gap = nn.AvgPool2d(kernel_size=53)
        self.fc2 = nn.Linear(16, 2)

        self.feature_map_shape = (16, 53, 53)
        self.out_features = 16

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv4(x))
        x = self.dropout_conv(x)
        return x

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_8_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_8_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 252 -> 126 -> 122 -> 122 -> 61 -> 57 -> 57 -> 53 -> 53
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)   
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)                      
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)                     
        self.conv3b = nn.Conv2d(12, 12, kernel_size=3, padding=1)                   
        self.conv4 = nn.Conv2d(12, 16, kernel_size=5, padding=0)                    
        self.conv4b = nn.Conv2d(16, 16, kernel_size=3, padding=1)                   

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(16, 2)

        self.feature_map_shape = (16, 53, 53)
        self.out_features = 16

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4b(x))
        x = self.dropout_conv(x)
        return x

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x


class DMTimeBinaryClassificator241002_9_GAP(BinaryClassifierBase):
    def __init__(self, resol, mode, dropout, device):
        super().__init__(resol, mode, dropout, device)

        if resol != 256:
            raise ValueError(
                f"DMTimeBinaryClassificator241002_9_GAP aktuell nur für resol=256 implementiert, bekommen: {resol}"
            )

        # 256 -> 252 -> 252 -> 126 -> 122 -> 122 -> 61 -> 57 -> 57 -> 53 -> 53 -> 49
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=5, padding=0)   
        self.conv1b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=0)                      
        self.conv2b = nn.Conv2d(8, 8, kernel_size=3, padding=1)                     
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                          
        self.conv3 = nn.Conv2d(8, 12, kernel_size=5, padding=0)                     
        self.conv3b = nn.Conv2d(12, 12, kernel_size=3, padding=1)                   
        self.conv4 = nn.Conv2d(12, 16, kernel_size=5, padding=0)                    
        self.conv4b = nn.Conv2d(16, 16, kernel_size=3, padding=1)                   
        self.conv5 = nn.Conv2d(16, 16, kernel_size=5, padding=0)                    

        self.gap = nn.AvgPool2d(kernel_size=49)
        self.fc2 = nn.Linear(16, 2)

        self.feature_map_shape = (16, 49, 49)
        self.out_features = 16

    def classifier_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4b(x))
        x = F.relu(self.conv5(x))
        x = self.dropout_conv(x)
        return x

    def pooled_features(self, x):
        x = self.classifier_features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        return x

    def classifier(self, x):
        x = self.pooled_features(x)
        x = self.fc2(x)
        return x