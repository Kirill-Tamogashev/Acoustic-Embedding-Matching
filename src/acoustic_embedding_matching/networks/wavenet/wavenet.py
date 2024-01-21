import torch
import torch.nn as nn

from .modules import WaveNetBlock, CausalConv1d


class WaveNet(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            hidden_channels: int = 128,
            cond_channels: int = 16,
            num_layers: int = 10
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.causal_convolution = CausalConv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )
        self.layer_list = nn.ModuleList([])
        for layer_index in range(num_layers):
            self.layer_list.append(
                WaveNetBlock(
                    hidden_channels=hidden_channels,
                    output_channels=output_channels,
                    cond_channels=cond_channels,
                    dilation=2 ** layer_index
                )
            )

        self.end_conv_1 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.end_conv_2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.size(0), self.hidden_channels, x.size(-1), device=x.device)
        x = self.causal_convolution(x)

        residual = x
        for idx, layer in enumerate(self.layer_list):
            residual, skip = layer(residual, condition)
            out += skip

        x = torch.relu(out)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
