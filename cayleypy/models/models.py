import torch
from attr import dataclass
from numba.core.ccallback import CFunc
from torch import nn

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration used to describe ML model."""

    model_type: str
    input_size: int
    num_classes_for_one_hot: int
    layers_sizes: list[int]

    @staticmethod
    def from_dict(cfg: dict[str, any]):
        """Creates config from Python dict."""
        return ModelConfig(
            model_type=cfg["model_type"],
            input_size=cfg["input_size"],
            num_classes_for_one_hot=cfg["num_classes_for_one_hot"],
            layers_sizes=cfg["layers_sizes"],
        )

    def create_model(self) -> nn.Module:
        """Creates model described by this config."""
        if self.model_type == "MLP":
            return MlpModel(self)
        else:
            raise ValueError("Unknown model type: " + self.model_type)


class MlpModel(nn.Module):
    """Multi-layer perceptron model."""

    def __init__(self, config):
        super(MlpModel, self).__init__()
        assert config.model_type == "MLP"
        self.num_classes_for_one_hot = config.num_classes_for_one_hot
        self.input_layer_size = config.input_size * self.num_classes_for_one_hot

        layers = []
        in_features = self.input_layer_size
        for hidden_dim in config.layers_sizes:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.one_hot(x.long(), num_classes=self.num_classes_for_one_hot).float().flatten(start_dim=-2)
        return self.layers(x).squeeze(-1)
