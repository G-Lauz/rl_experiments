import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation=torch.nn.ReLU):
        super(MLP, self).__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.mlp(x)
