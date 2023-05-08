import torch

class SimpleCNN(torch.nn.Module):

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_hidden_layers: int,
        use_batchnormalization: bool,
        num_classes: int,
        kernel_size: int = 3,
        activation_function: torch.nn.Module = torch.nn.ReLU()
        ):
    
            super().__init__()

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        pass