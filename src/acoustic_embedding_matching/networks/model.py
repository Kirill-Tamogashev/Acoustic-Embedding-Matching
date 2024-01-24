from typing import Optional

import torch
import torch.nn as nn

from torchaudio.transforms import Spectrogram

from src.acoustic_embedding_matching.networks.wavenet import WaveNet
from src.acoustic_embedding_matching.networks.bottleneck import Bottleneck
from src.acoustic_embedding_matching.networks.embedding_network import EmbeddingNetwork


class AcousticEmbeddingMatching(nn.Module):
    def __init__(
            self,
            input_channels: int = 1,
            output_channels: int = 1,
            hidden_channels: int = 128,
            embedding_size: int = 16,
            num_wavenet_layers: int = 10,
            num_embedding_layers: int = 5,
            bottleneck_channels: int = 16,
            bottleneck_kernel_size: int = 3,
            stft_kwargs: Optional[dict[str, int | bool]] = None,
    ) -> None:
        super().__init__()
        self.stft_kwargs = stft_kwargs if stft_kwargs is not None else {
            "n_fft": 1024, "win_length": 1024, "hop_length": 512
        }
        self.to_spectrogram = Spectrogram(**self.stft_kwargs)
        self.source_wavenet = WaveNet(
            input_channels=input_channels,
            output_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            cond_channels=embedding_size,
            num_layers=num_wavenet_layers,
        )
        self.target_wavenet = WaveNet(
            input_channels=bottleneck_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            cond_channels=embedding_size,
            num_layers=num_wavenet_layers,
        )
        self.bottleneck = Bottleneck(
            bottleneck_channels=bottleneck_channels,
            kernel_size=bottleneck_kernel_size,
        )
        self.embedding_network = EmbeddingNetwork(
            embedding_size=embedding_size,
            num_layers=num_embedding_layers,
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Transform audio from source acoustic domain to target acoustic domain.
        Args:
            source [Batch, Time]: wave tensor representing audio from source domain
            target [Batch, Time]: wave tensor represented audio from target domain

        Returns:
            Audio from source domain converted to target domain
        """

        source_embed = self.embedding_network(self.to_spectrogram(source))
        target_embed = self.embedding_network(self.to_spectrogram(target))

        hidden = self.source_wavenet(source.unsqueeze(1), source_embed)
        hidden = self.bottleneck(hidden)
        prediction = self.target_wavenet(hidden, target_embed)

        return prediction.squeeze(1)

    @torch.no_grad()
    def inference(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.eval()
        prediction = self(source, target)
        return prediction

