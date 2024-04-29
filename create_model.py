from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur


class SimpleModel(nn.Module):
    """A simple example torch model containing only a gaussian blur"""

    def __init__(self):
        super().__init__()
        self.transform = GaussianBlur(kernel_size=5, sigma=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.transform(x)
            return y


def create_model(model_path: Path):
    """Create and save an example jit model"""
    model = SimpleModel()
    example_input = torch.rand(1, 255, 255)
    jit_model = torch.jit.trace(model, example_inputs=example_input)
    print(f'Saving model to: {model_path.absolute()}')
    torch.jit.save(jit_model, model_path)


if __name__ == "__main__":
    model_path = Path(__file__).parent / "resources/model.pth"
    create_model(model_path)
