import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding=0, stride=1):
        super(ConvLayer, self).__init__()
        
        assert len(in_channels) == len(out_channels) == len(kernel_sizes), "Mismatch in the length of layer parameters"
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ) for in_c, out_c, kernel in zip(in_channels, out_channels, kernel_sizes)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PhaseAssignment(nn.Module):
    def __init__(self, window_size, num_phase):
        super(PhaseAssignment, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(window_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(50, num_phase)
        )
    
    def forward(self, x):
        return self.classifier(x)


class CaRaCountModel(nn.Module):
    def __init__(self, num_phase, window_size, 
                 kernel_sizes=[(1,7), (1,5), (1,3)], 
                 channels_in=[6, 64, 64], 
                 channels_out=[64, 64, 64]):
        super(CaRaCountModel, self).__init__()
        
        self.conv_layer = ConvLayer(channels_in, channels_out, kernel_sizes)
        
        # Compute the final feature size after convolutions
        feature_size = window_size
        for _, k in kernel_sizes:
            feature_size = (feature_size - k + 2 * 0) // 1 + 1
        
        self.gru = nn.GRU(feature_size * channels_out[-1], 200, num_layers=2, batch_first=True, dropout=0.5)
        self.classifier = PhaseAssignment(200, num_phase + 1)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1) 
        x, _ = self.gru(x)
        x = self.classifier(x)
        return x