from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from accelerate import Accelerator
import wandb


class BaseTrainer(ABC):
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            train_loader: torch.data.DataLoader,
            val_loader: torch.data.DataLoader,
            num_epochs: int = 100,
            iter_log_freq: int = 10_000,
            max_train_iterations: int = int("inf")
    ) -> None:
        self.num_epochs = num_epochs
        self.iter_log_freq = iter_log_freq
        self.max_train_iterations = max_train_iterations

        self.accelerator = Accelerator()

        model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            [model, optimizer, train_loader, scheduler]
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self) -> None:
        with wandb.init() as logger:
            total_iterations = 0

            for epoch in range(self.num_epochs):
                for train_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    loss, train_batch_data = self.train_step(train_batch, epoch, total_iterations)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()

                    total_iterations += 1
                    if total_iterations % self.iter_log_freq == 0:
                        self.train_log(logger, train_batch_data, epoch, total_iterations)

                    if total_iterations >= self.max_train_iterations:
                        return

                with torch.no_grad():
                    for val_batch in self.val_loader:
                        loss, val_batch_data = self.val_step(val_batch, epoch, total_iterations)
                self.val_log(logger, val_batch_data, epoch, total_iterations)

    @abstractmethod
    def train_step(self, batch, epoch, iteration):
        raise NotImplemented("Implement for a specific model")

    @abstractmethod
    def val_step(self, batch, epoch, iteration):
        raise NotImplemented("Implement for a specific model")

    @abstractmethod
    def val_log(self, logger, log_data, epoch, iteration):
        raise NotImplemented("Implement for a specific model")

    @abstractmethod
    def train_log(self, logger, log_data, epoch, iteration):
        raise NotImplemented("Implement for a specific model")