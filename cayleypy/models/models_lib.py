"""Library of pretrained models."""

from .models import ModelConfig

MODELS = {
    "lrx-32": ModelConfig(
        model_type="MLP",
        input_size=32,
        num_classes_for_one_hot=32,
        layers_sizes=[1024, 1024, 1024],
        weights_kaggle_id="fedimser/lrx-32-by-mrnnnn/PyTorch/model_final/1",
        weights_path="model_final.pth",
    )
}
