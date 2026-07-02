import torch
import torch.nn as nn

class BinaryClassifierBase(nn.Module):
    def __init__(self, resol, mode, dropout, device):
        super().__init__()

        self.mode = mode
        self.dropout = dropout
        self.resol = resol
        self.device = device

        if self.mode not in {"dmt", "ft", "dmft"}:
            raise AttributeError(
                f"mode {mode} currently not supported. Use 'dmt', 'ft' or 'dmft' for selecting input mode"
            )

        if self.mode in {"dmt", "ft"}:
            self.input_channels = 1
        else:  # "dmft"
            self.input_channels = 2

        dropout_conv = nn.Dropout2d(p=0.2)
        dropout_fc = nn.Dropout(p=0.4)
        if not dropout:
            dropout_conv = nn.Identity()
            dropout_fc = nn.Identity()

        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc

        if self.mode == "dmt":
            self._prepare_input = self._prepare_input_dmt
        elif self.mode == "ft":
            self._prepare_input = self._prepare_input_ft
        else:  # "dmft"
            self._prepare_input = self._prepare_input_dmft

    def _prepare_input_dmt(self, batch):
        x = batch['dm_time'].to(self.device, non_blocking=True)
        return x.unsqueeze(1)

    def _prepare_input_ft(self, batch):
        x = batch["freq_time"].to(self.device, non_blocking=True)
        return x.unsqueeze(1)

    def _prepare_input_dmft(self, batch):
        x_ft = batch["freq_time"].to(self.device, non_blocking=True).unsqueeze(1)
        x_dmt = batch["dm_time"].to(self.device, non_blocking=True).unsqueeze(1)
        return torch.cat((x_ft, x_dmt), dim=1)
    
    def features(self, x):
        return self._prepare_input(x)
    
    def classifier(self, x):
        pass
    
    def classifier_features(self, x):
        pass
    
    def forward(self, x):
        x = self._prepare_input(x)
        return self.classifier(x)
    
    def forward_features(self, x):
        x = self._prepare_input(x)
        return self.classifier_features(x)
