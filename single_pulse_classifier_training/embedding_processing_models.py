import torch.nn as nn
import torch.nn.functional as F

from ResNet import BasicBlock

class ResNetEmbeddingProcessing(nn.Module):
    # Small ResNet variant for the feature maps
    def __init__(self, in_channels=12, num_blocks=[1, 1], base_channels=32, pool_size=5, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.in_planes = base_channels
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Create blocks
        self.layer1 = self._make_layer(BasicBlock, base_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, num_blocks[1], stride=2)
        
        flattened_dim = base_channels * 2 * pool_size * pool_size

        self.pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, dropout_prob=self.dropout))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, dropout_prob=self.dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out)
        return self.mlp(out)


class ConvGAPEmbeddingProcessing(nn.Module):
    #CNN head on f_small feature maps; (B, C, H, W) -> logits

    def __init__(self, in_channels=12, cnn_channels=64, extra_conv=True, dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]

        if extra_conv:
            layers.extend([
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])

        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels, 2),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvMLPEmbeddingProcessing(nn.Module):
    #CNN head on raw f_small feature maps, then spatial pooling and one hidden MLP layer

    def __init__(self, in_channels=12, cnn_channels=64, extra_conv=True, pool_size=5, hidden_dim=256, dropout=0.0):
        super().__init__()
        flattened_dim = cnn_channels * pool_size * pool_size

        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]

        if extra_conv:
            layers.extend([
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])

        layers.extend([
            nn.AdaptiveMaxPool2d((pool_size, pool_size)),
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpatialPoolMLPEmbeddingProcessing(nn.Module):
    #Pool raw f_small feature maps to k x k; classify with one hidden MLP layer

    def __init__(self, in_channels=12, pool_size=5, hidden_dim=256, dropout=0.0, pool_type="max"):
        super().__init__()
        flattened_dim = in_channels * pool_size * pool_size

        if pool_type == "max":
            pool_layer = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        elif pool_type == "avg":
            pool_layer = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        else:
            raise ValueError(f"Unknown pool_type '{pool_type}'. Use 'max' or 'avg'.")

        self.net = nn.Sequential(
            pool_layer,
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class PooledFeatureMLPEmbeddingProcessing(nn.Module):
    #MLP head on f_small.pooled_features output; (B, C) -> logits

    def __init__(self, in_features=12, hidden_dim=256, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class PooledFeatureMLP12864EmbeddingProcessing(nn.Module):
    #MLP on f_small.pooled_features: 12 -> 128 -> 64 -> 2; just testing for larger models

    def __init__(self, in_features=12, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class PooledFeatureMLP256256EmbeddingProcessing(nn.Module):
    #MLP on f_small.pooled_features: 12 -> 256 -> 256 -> 2; just testing for larger models

    def __init__(self, in_features=12, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


class DeepConvMLPEmbeddingProcessing(nn.Module):
    # Tiefere Variante des ConvMLP fuer komplexere Feature-Maps (z.B. von f_mid)
    
    def __init__(self, in_channels=12, cnn_channels=64, num_conv_blocks=3, pool_size=5, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        # Stapeln von Convolutional Blocks, um komplexe Features zu extrahieren
        for i in range(num_conv_blocks):
            out_ch = cnn_channels * (2 ** i) if i < 2 else cnn_channels * 2 # bspw. 64 -> 128 -> 128
            layers.extend([
                nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch), # WICHTIG für tiefere Netze 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            current_channels = out_ch
            
        flattened_dim = current_channels * pool_size * pool_size

        layers.extend([
            nn.AdaptiveMaxPool2d((pool_size, pool_size)),
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_embedding_processing(model_name, in_channels=12, cnn_channels=64, extra_conv=True, pool_size=5, hidden_dim=256, dropout=0.0, pool_type="max"):
    #Create an embedding-processing head; tell EmbeddingRejector which f_small feature hook to use

    if model_name == "resnet_small":
        return (
            ResNetEmbeddingProcessing(
                in_channels=in_channels,
                num_blocks=[1, 1],
                base_channels=cnn_channels if cnn_channels >= 32 else 32, # Ensure at least 32
                pool_size=pool_size,
                hidden_dim=hidden_dim,
                dropout=dropout,
            ),
            "classifier_features",
        )

    if model_name == "conv_gap":
        return (
            ConvGAPEmbeddingProcessing(
                in_channels=in_channels,
                cnn_channels=cnn_channels,
                extra_conv=extra_conv,
                dropout=dropout,
            ),
            "classifier_features",
        )

    if model_name == "conv_mlp":
        return (
            ConvMLPEmbeddingProcessing(
                in_channels=in_channels,
                cnn_channels=cnn_channels,
                extra_conv=extra_conv,
                pool_size=pool_size,
                hidden_dim=hidden_dim,
                dropout=dropout,
            ),
            "classifier_features",
        )

    if model_name == "spatial_pool_mlp":
        return (
            SpatialPoolMLPEmbeddingProcessing(
                in_channels=in_channels,
                pool_size=pool_size,
                hidden_dim=hidden_dim,
                dropout=dropout,
                pool_type=pool_type
            ),
            "classifier_features",
        )

    if model_name == "pooled_mlp":
        return (
            PooledFeatureMLPEmbeddingProcessing(
                in_features=in_channels,
                hidden_dim=hidden_dim,
                dropout=dropout,
            ),
            "pooled_features",
        )

    if model_name == "pooled_mlp_128_64":
        return (
            PooledFeatureMLP12864EmbeddingProcessing(
                in_features=in_channels,
                dropout=dropout,
            ),
            "pooled_features",
        )

    if model_name == "pooled_mlp_256_256":
        return (
            PooledFeatureMLP256256EmbeddingProcessing(
                in_features=in_channels,
                dropout=dropout,
            ),
            "pooled_features",
        )

    if model_name == "deep_conv_mlp":
        return (
            DeepConvMLPEmbeddingProcessing(
                in_channels=in_channels,
                cnn_channels=cnn_channels,
                num_conv_blocks=3,
                pool_size=pool_size,
                hidden_dim=hidden_dim,
                dropout=dropout,
            ),
            "classifier_features",
        )

    raise ValueError(
        f"Unknown embedding processing model '{model_name}'. "
        "Use one of: conv_gap, conv_mlp, spatial_pool_mlp, pooled_mlp, pooled_mlp_128_64, pooled_mlp_256_256."
    )
