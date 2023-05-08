import torch
import torch.nn as nn

class SimpleNetwork(torch.nn.Module):
    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int,
                  activation_function: torch.nn.Module = torch.nn.ReLU() ):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_neurons, out_features=hidden_neurons)
        self.layer_2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.layer_3 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.layer_4 = nn.Linear(in_features=hidden_neurons, out_features=output_neurons)
        self.activation_function = activation_function


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        self.activation_function(x)
        x = x = self.layer_2(x)
        self.activation_function(x)
        x = self.layer_3(x)
        self.activation_function(x)
        output = self.layer_4(x)
        return output


if __name__ == "__main__":
    torch.random.manual_seed(0)
    simple_network = SimpleNetwork(10, 20, 5)
    input = torch.randn(1, 10)
    output = simple_network(input)
    print(output)