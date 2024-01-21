import torch
import torch.nn as nn


class DilatedConditionalConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cond_channels: int,
            dilation: int,
            kernel_size: int = 2
    ) -> None:
        super().__init__()
        self.conditional_conv = nn.Conv1d(
            in_channels=cond_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.dilated_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation,
            stride=1, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
            Dilated convolution with optional condition
        Args:
            x [Batch, Channels, Time]: hidden state of waveform
            condition [Batch, EmbeddingSize]: acoustic embedding

        Returns:
            Conditional convolution output.
        """
        condition.unsqueeze_(-1)
        condition = self.conditional_conv(condition)
        condition = torch.tile(condition, (1, 1, x.size(2)))
        return self.dilated_conv(x) + condition


class CausalConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 2,
            padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding,
                              stride=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x[:, :, :-1]


class WaveNetBlock(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            output_channels: int,
            cond_channels: int,
            dilation: int,
            kernel_size: int = 2,
    ) -> None:
        super().__init__()

        self.filter_conv = DilatedConditionalConv1d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            cond_channels=cond_channels,
            dilation=dilation,
            kernel_size=kernel_size,
        )
        self.gate_conv = DilatedConditionalConv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            cond_channels=cond_channels,
            dilation=dilation,
            kernel_size=kernel_size,
        )
        self.skip_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=False,
        )
        self.res_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        filter_ = torch.tanh(self.filter_conv(x, condition))
        gate = torch.sigmoid(self.gate_conv(x, condition))
        x = gate * filter_

        residual = self.res_conv(x) + x
        skip = self.skip_conv(x)
        return residual, skip

