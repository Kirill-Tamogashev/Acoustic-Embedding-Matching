from abc import ABC

import torch
import torch.nn as nn

import wandb
from .train_base import BaseTrainer


class AcousticMatchingTrainer(BaseTrainer, ABC):
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            train_loader: torch.data.DataLoader,
            val_loader: torch.data.DataLoader,
            criterion: torch.nn.Module,
            num_epochs: int = 100,
            log_data_freq: int = 100_000,
            log_loss_freq: int = 1_000,
            max_train_iterations: int = int("inf"),
            sample_rate: int = 16000,
            log_num_train_objects: int = 5,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            log_data_freq=log_data_freq,
            log_loss_freq=log_loss_freq,
            max_train_iterations=max_train_iterations,
        )
        self.log_num_train_objects = log_num_train_objects
        self.sample_rate = sample_rate
        self.criterion = criterion

    def train_step(self, batch, epoch, iteration):
        source, condition, target = batch
        prediction = self.model(source, condition)
        loss = self.criterion(prediction, target)
        return loss, [loss.item(), prediction, target, source, condition]

    def train_log(self, logger, log_data, epoch, iteration):
        _, prediction, target, source, condition = log_data

        for i in range(self.log_num_train_objects):
            data = [prediction[i], target[i], source[i], condition[i]]
            logger.log(self.generate_log_dict("train", data))

    @torch.no_grad()
    def val_step(self, batch, epoch, iteration):
        self.model.eval()
        loss, batch_data = self.train_step(batch, epoch, iteration)
        self.model.train()
        return loss, batch_data

    def val_log(self, logger, log_data, epoch, iteration):
        self.compute_metrics(log_data)
        one_batch = log_data[0]
        for i in range(self.log_num_train_objects):
            data = [one_batch[i], one_batch[i], one_batch[i], one_batch[i]]
            logger.log(self.generate_log_dict("val", data))

    def generate_log_dict(self, stage: str, data: list[torch.Tensor]) -> dict:
        loss, prediction, target, source, condition = data
        return {
            f"{stage}/predictions":             self._log_audio(prediction, "Predicted audio"),
            f"{stage}/source":                  self._log_audio(source, "Source audio"),
            f"{stage}/target":                  self._log_audio(target, "Target audio"),
            f"{stage}/condition":               self._log_audio(condition, "Condition audio"),
            f"{stage}/prediction spectrogram":  self._log_spectrogram(prediction, "Predicted spectrogram"),
            f"{stage}/source spectrogram":      self._log_spectrogram(prediction, "Source spectrogram"),
            f"{stage}/target spectrogram":      self._log_spectrogram(prediction, "Target spectrogram"),
            f"{stage}/condition spectrogram":   self._log_spectrogram(prediction, "Condition spectrogram"),
        }

    @staticmethod
    def _log_spectrogram(audio: torch.Tensor, caption: str):
        spectrogram = torch.stft(audio, n_fft=1024, hop_length=512, return_complex=True).real
        spectrogram = spectrogram.cpu().numpy()
        return wandb.Image(spectrogram, caption=caption)

    def _log_audio(self, audio: torch.Tensor, caption: str):
        return wandb.Audio(audio.cpu().numpy(), caption=caption, sample_rate=self.sample_rate)

    def compute_metrics(self, data):
        pass
