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
            iter_log_freq: int = 10_000,
            max_train_iterations: int = int("inf"),
            sample_rate: int = 16000,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            iter_log_freq=iter_log_freq,
            max_train_iterations=max_train_iterations,
        )
        self.sample_rate = sample_rate
        self.criterion = criterion

    def train_step(self, batch, epoch, iteration):
        source, condition, target = batch
        prediction = self.model(source, condition)
        loss = self.criterion(prediction, target)
        return loss, [loss, prediction, target, source, condition]

    def train_log(self, logger, log_data, epoch, iteration):
        loss, prediction, target, source, condition = log_data

        logger.log({"train/train loss": loss})
        # for
        logger.log(
            {
                "train/predictions": wandb.Audio(prediction.cpu().numpy(),
                                                 caption="Predicted audio", sample_rate=self.sample_rate),
                "train/source":     wandb.Audio(prediction.cpu().numpy(),
                                                caption="Source audio", sample_rate=self.sample_rate),
                "train/target":     wandb.Audio(prediction.cpu().numpy(),
                                                caption="Target Audio", sample_rate=self.sample_rate),
                "train/condition":  wandb.Audio(prediction.cpu().numpy(),
                                                caption="Condition", sample_rate=self.sample_rate),
            }
        )


    def val_step(self, batch, epoch, iteration):
        source, condition, target = batch

        pass

    def val_log(self, logger, log_data, epoch, iteration):
        pass

    def log_spectrogram(self, ):
        pass
    def _log_audio(self, ):
        pass
