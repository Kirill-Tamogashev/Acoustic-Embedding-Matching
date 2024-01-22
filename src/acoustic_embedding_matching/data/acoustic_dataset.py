from random import choices
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.functional import fftconvolve


def split_audio(audio: torch.Tensor, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(len(audio), max_length)
    length = length - 1 if length % 2 == 1 else length
    return audio[:length // 2], audio[length // 2:]


class AcousticDataset(Dataset):
    def __init__(
            self,
            audio_path: str | Path,
            ir_path: str | Path,
            sample_rate: int = 16_000,
            max_length: int = 64_000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_length = max_length

        self.audio_path = audio_path if isinstance(audio_path, str) else Path(audio_path)
        self.ir_path = ir_path if isinstance(ir_path, str) else Path(ir_path)

        self.audios = [path for path in self.audio_path.glob(".wav")]
        self.impulse_responses = [path for path in self.ir_path.glob(".wav")]

    def __len__(self):
        return len(self.audio_path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        audio, sample_rate = torchaudio.load(self.audios[index])

        ir_path_1, ir_path_2 = choices(self.impulse_responses)
        impulse_response_1, _ = torchaudio.load(ir_path_1)
        impulse_response_2, _ = torchaudio.load(ir_path_2)

        audio_1, audio_2 = split_audio(audio, max_length=self.max_length)

        sample = fftconvolve(audio_1, impulse_response_1, mode="same")
        sample_plus = fftconvolve(audio_2, impulse_response_1, mode="same")
        sample_minus = fftconvolve(audio_1, impulse_response_2, mode="same")

        return sample, sample_plus, sample_minus

    def augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        pass
