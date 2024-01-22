from abc import ABC

import torch
import torch.nn as nn

from .train_base import BaseTrainer


class EmbeddingNetworkTrainer(BaseTrainer, ABC):
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
            max_train_iterations: int = int("inf")
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
        self.criterion = criterion

    def train_step(self, batch, epoch, iteration):
        inputs, plus_sample, minus_sample = batch

        embedding = self.model(inputs)
        plus_embedding = self.model(plus_sample)
        minus_embedding = self.model(minus_sample)

        loss = self.criterion(embedding, plus_embedding, minus_embedding)
        return loss, []

    def val_step(self, batch, epoch, iteration):
        inputs, plus_sample, minus_sample = batch

        self.model.eval()
        embedding = self.model(inputs)
        plus_embedding = self.model(plus_sample)
        minus_embedding = self.model(minus_sample)
        self.model.train()

        loss = self.criterion(embedding, plus_embedding, minus_embedding)
        return loss, []

    def val_log(self, logger, log_data, epoch, iteration):
        pass

    def train_log(self, logger, log_data, epoch, iteration):
        pass


