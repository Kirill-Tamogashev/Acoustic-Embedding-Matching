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
            iter_log_freq: int = 10_000,
            max_train_iterations: int = int("inf")
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
        self.criterion = criterion

    def train_step(self, batch, epoch, iteration):
        source, condition, target = batch
        prediction = self.model(source, condition)
        loss = self.criterion(prediction, target)
        return loss, _

