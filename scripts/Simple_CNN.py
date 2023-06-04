import torch
import torch.nn as nn

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

        self.activation_function = activation_function
        padding = (kernel_size - 1) // 2

        hidden_layers = []

        for _ in range(num_hidden_layers):
            layer = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding)
            hidden_layers.append(layer)

            if use_batchnormalization:
                hidden_layers.append(nn.BatchNorm2d(hidden_channels))

            hidden_layers.append(activation_function)
            input_channels = hidden_channels

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_channels * 64 * 64, num_classes)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        output = self.hidden_layers(input_images)
        output = torch.flatten(output, start_dim=1)
        output = self.output_layer(output)
        return output

if __name__ == "__main__":
    torch.random.manual_seed(0)
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(output)
    