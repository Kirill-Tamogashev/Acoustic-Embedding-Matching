import torch

import torch.nn.functional as f


class ContrastiveLoss:
    def __call__(self, x, x_plus, x_minus):
        mse_plus = f.mse_loss(x_plus, x, reduction='none')
        mse_minus = f.mse_loss(x_minus, x, reduction='none')

        return - torch.log1p(torch.exp(mse_minus.sum(dim=-1) - mse_plus.sum(dim=-1))).mean()


class STFTLoss:
    def __init__(
            self,
            metric: str = "l1",
            win_lengths: tuple[int, ...] = (2048, 512),
            hop_lengths: tuple[int, ...] = (512, 128)
    ) -> None:
        self.metric = f.l1_loss if metric == "l1" else f.mse_loss
        self.win_lengths = win_lengths
        self.hop_lengths = hop_lengths

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.as_tensor(0)
        for (win_length, hop_length) in zip(self.win_lengths, self.hop_lengths):
            pred = torch.stft(prediction, n_fft=win_length, hop_length=hop_length, return_complex=True)
            true = torch.stft(target, n_fft=win_length, hop_length=hop_length, return_complex=True)
            loss += self.metric(pred, true, reduction="none")

        return loss.view(loss.size(0), -1).sum(dim=1).mean()
