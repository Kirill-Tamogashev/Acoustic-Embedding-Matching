import torch
import torch.nn as nn
import torch.nn.functional as f


class Conv2dBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            max_pool_kernel: tuple[int, int],
            kernel_size: tuple[int, int] = (5, 5),
            dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding="same")
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_kernel)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class EmbeddingNetwork(nn.Module):
    def __init__(
            self,
            embedding_size: int = 16,
            num_layers: int = 5,
    ):
        super().__init__()
        self.pre_network = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, padding="same")

        self.layer_list = nn.ModuleList()
        for layer_index in range(num_layers):
            self.layer_list.extend(
                [
                    Conv2dBlock(32 * 2 ** layer_index, 32 * 2 ** (layer_index + 1), max_pool_kernel=(1, 2)),
                    Conv2dBlock(32 * 2 ** (layer_index + 1), 32 * 2 ** (layer_index + 1), max_pool_kernel=(2, 1))
                ]
            )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward call for the Embedding Network.

            Args:
                x: [Batch, Length, FreqDim]

            Returns:
                embedding: [Batch, EmbeddDim]
        """

        x = self.adjust_shape(x)
        x = self.pre_network(x)

        for layer in self.layer_list:
            x = layer(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = self.avg_pool(x)
        x.squeeze_(2)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        embedding = f.normalize(self.fc3(x), p=2.0, dim=1)

        return embedding

    @staticmethod
    def adjust_shape(x):
        if len(x.size()) == 3:
            return x.unsqueeze(1)
        elif len(x.size() == 4):
            return x
        else:
            raise AssertionError("Input size is incorrect")


