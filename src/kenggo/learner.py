"""Compact, history-free population policy and recorded-outcome value model."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kenggo.encoding import INPUT_KEYS, SCALARS, array_schema
from kenggo.go import PASS, SIZE
from kenggo.kgs_source import RANKS


@dataclass(frozen=True)
class ModelConfig:
    blocks: int = 8
    channels: int = 128
    rank_conditioned: bool = True

    def __post_init__(self):
        if type(self.blocks) is not int or not 1 <= self.blocks <= 16:
            raise ValueError("Use between 1 and 16 residual blocks")
        if type(self.channels) is not int or not 16 <= self.channels <= 256 or self.channels % 8:
            raise ValueError("Use 16 to 256 channels in multiples of eight")
        if type(self.rank_conditioned) is not bool:
            raise ValueError("Rank conditioning must be a boolean")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, board):
        residual = F.relu(self.norm1(self.conv1(board)))
        return F.relu(board + self.norm2(self.conv2(residual)))


class PopulationModel(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("PopulationModel requires a ModelConfig")
        self.config = config
        channels = config.channels
        # Index zero is learned, including when both players have unknown ranks.
        self.rank_embedding = nn.Embedding(len(RANKS) + 1, 16)
        self.metadata = nn.Sequential(nn.Linear(2 * len(SCALARS) + 34, channels), nn.ReLU(),
                                      nn.Linear(channels, channels))
        self.stem = nn.Conv2d(4, channels, 3, padding=1, bias=False)
        self.stem_norm = nn.GroupNorm(8, channels)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(config.blocks)))
        self.board_policy = nn.Conv2d(channels, 1, 1)
        self.pass_policy = nn.Linear(channels, 1)
        self.value = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1), nn.Tanh())

    def forward(self, inputs):
        """Accept batched shared inputs; normalization belongs to the caller."""
        if set(inputs) != set(INPUT_KEYS) or any(not isinstance(v, torch.Tensor) for v in inputs.values()):
            raise ValueError("Model inputs must contain only the shared encoding tensors")
        board = inputs["board"]
        if board.ndim != 4 or board.shape[0] == 0:
            raise ValueError("Model inputs require a nonempty batch")
        batch = board.shape[0]
        dtypes = {"bool": torch.bool, "float32": torch.float32, "int16": torch.int16, "int8": torch.int8}
        for name, (dtype, shape) in array_schema(0).items():
            tensor = inputs[name]
            if tensor.shape != (batch,) + shape or tensor.dtype != dtypes[dtype] or tensor.device != board.device:
                raise ValueError(f"Invalid history-free input shape, dtype or device: {name}")
        legal = inputs["legal"]
        if not legal.any(dim=1).all():
            raise ValueError("Every policy observation must have a legal action")
        scalar_valid = inputs["scalar_valid"]
        scalars = torch.where(scalar_valid, inputs["scalars"], 0)
        if not torch.isfinite(scalars).all():
            raise ValueError("Visible metadata must be finite")
        rank_valid = inputs["rank_valid"] if self.config.rank_conditioned else torch.zeros_like(inputs["rank_valid"])
        ranks = torch.where(rank_valid, inputs["ranks"], 0).long()
        if ((ranks[rank_valid] < 1) | (ranks[rank_valid] > len(RANKS))).any():
            raise ValueError("Visible ranks must use the shared rank vocabulary")
        metadata = torch.cat((scalars, scalar_valid.float(), self.rank_embedding(ranks).flatten(1),
                              rank_valid.float()), dim=1)
        planes = torch.cat((board.float(), legal[:, :PASS].reshape(batch, 1, SIZE, SIZE).float()), dim=1)
        trunk = F.relu(self.stem_norm(self.stem(planes) + self.metadata(metadata)[:, :, None, None]))
        trunk = self.blocks(trunk)
        pooled = trunk.mean(dim=(2, 3))
        logits = torch.cat((self.board_policy(trunk).flatten(1), self.pass_policy(pooled)), dim=1)
        return logits.masked_fill(~legal, -torch.inf), self.value(pooled).squeeze(1)
