import torch


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)
