from random import choices, choice
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F


def sample_uniform(low: int | float, high: int | float, size: tuple[int, ...] | int = 1) -> torch.Tensor:
    low, high = int(low), int(high)
    u = torch.rand(size)
    return low * (1 - u) + high * u


def split_audio(audio: torch.Tensor, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(len(audio), max_length)
    length = length - 1 if length % 2 == 1 else length
    return audio[:length // 2], audio[length // 2:]


def drr_augmentation(ir: torch.Tensor, sample_rate: int, drr: float,  t_0: float = 2.5) -> torch.Tensor:
    """
        Augment DRR of the given ir.

        This function changes DRR of the given Impulse Response (IR)
        following the method, described in `https://arxiv.org/pdf/1909.03642.pdf`

    Args:
        ir: [IR_Length] - input IR
        sample_rate: sample rate
        drr: desired DRR value
        t_0: tolerance window

    Returns:
        augmented IR
    """
    t_d = ir.abs().max(dim=-1).item()
    t_0 = sample_rate * 1e-3 * t_0
    mask = torch.zeros_like(ir)
    mask[int(t_d - t_0): int(t_d + t_0)] = 1

    h_e = ir * mask
    h_l = ir * (1 - mask)

    w = torch.hann_window(ir.size(-1))

    a = torch.sum((w * h_e) ** 2)
    b = 2 * torch.sum(w * (1 - w) * h_e ** 2)
    c = torch.sum(((1 - w) * h_e) ** 2) - 10 ** (drr / 10) * torch.sum(h_l ** 2)

    alpha = (-b + torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return h_l + alpha * w * h_e + (1 - w) * h_e


def multi_band_filter(
        audio: torch.Tensor,
        sample_rate: int,
        central_frequencies: list[int] | list[torch.Tensor] | torch.Tensor,
        gains: list[int] | list[torch.Tensor] | torch.Tensor = None,
) -> torch.Tensor:
    if gains is None:
        gains = [0] * len(central_frequencies)

    assert len(gains) == len(central_frequencies), "The lengths of gains and central_frequencies differ!"

    for freq, gain in zip(central_frequencies, gains):
        audio = F.bandpass_biquad(audio, sample_rate, freq)
        audio = F.gain(audio, gain)
    return audio


class AcousticDataset(Dataset):
    def __init__(
            self,
            audio_path: str | Path,
            ir_path: str | Path,
            noise_path: str | Path,
            sample_rate: int = 16_000,
            max_length: int = 64_000,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.max_length = max_length

        self.audio_path = audio_path if isinstance(audio_path, str) else Path(audio_path)
        self.ir_path = ir_path if isinstance(ir_path, str) else Path(ir_path)
        self.noise_path = noise_path if isinstance(noise_path, str) else Path(noise_path)

        self.audios = [path for path in self.audio_path.glob(".wav")]
        self.impulse_responses = [path for path in self.ir_path.glob(".wav")]
        self.noises = [path for path in self.noise_path.glob(".wav")]

    def __len__(self):
        return len(self.audio_path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        audio, sample_rate = torchaudio.load(self.audios[index])

        ir_path_1, ir_path_2 = choices(self.impulse_responses)
        ir_1, _ = torchaudio.load(ir_path_1)
        ir_1 = self.augment_impulse_response(ir_1)
        ir_2, _ = torchaudio.load(ir_path_2)
        ir_2 = self.augment_impulse_response(ir_2)

        audio_1, audio_2 = split_audio(audio, max_length=self.max_length)
        audio_1 = self.augment_audio(audio_1)
        audio_2 = self.augment_audio(audio_2)

        sample = F.fftconvolve(audio_1, ir_1, mode="same")
        sample_plus = F.fftconvolve(audio_2, ir_1, mode="same")
        sample_minus = F.fftconvolve(audio_1, ir_2, mode="same")

        return sample, sample_plus, sample_minus

    def resample_audio(self, audio: torch.Tensor) -> torch.Tensor:
        new_sample_rate = sample_uniform(0.9 * self.sample_rate, 1.1 * self.sample_rate, 1)
        gain = sample_uniform(0.5, 1.5)
        return F.resample(audio, self.sample_rate, new_sample_rate.item()) * gain

    def add_random_noise(self, audio: torch.Tensor) -> torch.Tensor:
        noise = torchaudio.load(choice(self.noises))
        noise = multi_band_filter(noise, self.sample_rate, [30, 250, 800, 2300])
        snr = sample_uniform(10, 30)
        return F.add_noise(audio, noise, snr.item())

    def augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio = self.resample_audio(audio)
        audio = self.add_random_noise(audio)
        return audio

    def augment_impulse_response(self, ir: torch.Tensor) -> torch.Tensor:
        ir = drr_augmentation(ir, self.sample_rate, ...)
        ir = F.resample(ir, self.sample_rate, ...)

        freqs = [sample_uniform(0, 50), sample_uniform(50, 300),
                 sample_uniform(300, 1500), sample_uniform(1500, 8000)]
        gains = sample_uniform(-10, 10, 4)

        ir = multi_band_filter(ir, self.sample_rate, freqs, gains)
        return ir
