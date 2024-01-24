import torch
import torch.nn as nn

from src.acoustic_embedding_matching.networks.wavenet.modules import CausalConv1d


class Bottleneck(nn.Module):
    def __init__(
            self,
            bottleneck_channels: int = 16,
            kernel_size: int = 3,
            n_layers: int = 3
    ) -> None:
        super().__init__()

        self.conv_list = nn.ModuleList([])
        for _ in range(n_layers):
            self.conv_list.append(
                CausalConv1d(bottleneck_channels, bottleneck_channels, kernel_size,
                             front_padding=kernel_size - 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_list:
            x = torch.relu(conv(x))
        return x
