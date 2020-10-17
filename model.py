import torch
import torch.nn as nn
from typing import Tuple

from math import sqrt


class EnsembleFNN(nn.Module):
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dims_hidden_neurons: Tuple = (64, 64),
                 num_nets: int = 10,  # number of nets
                 activation=torch.relu,
                 seed=0,
                 ):
        super(EnsembleFNN, self).__init__()

        torch.manual_seed(seed)

        self.num_nets = num_nets
        self.activation = activation

        self.weights = []
        self.bias = []

        n_neurons = (dim_input, ) + dims_hidden_neurons + (dim_output, )
        for ii, (dim_in, dim_out) in enumerate(zip(n_neurons[:-1], n_neurons[1:])):
            weight = nn.Parameter(torch.randn(num_nets, dim_in, dim_out) * sqrt(2 / (dim_in + dim_out)),
                                  requires_grad=True)  # Xavier Initialization
            bias = nn.Parameter(torch.zeros(1, num_nets, dim_out, requires_grad=True),
                                requires_grad=True)  # 1 is for broadcasting
            self.weights.append(weight)
            self.bias.append(bias)

        self.weights = nn.ParameterList(self.weights)
        self.bias = nn.ParameterList(self.bias)

        self.num_layers = len(self.weights)

    def forward(self, input: torch.Tensor):
        x = input
        if len(x.shape) == 2:
            # for input shape (batch, input_feature), pass the same input through all nets
            x = torch.einsum('bi,nio->bno', x, self.weights[0]) + self.bias[0]
            x = self.activation(x)
            for ii in range(1, self.num_layers-1):
                x = torch.einsum('bni,nio->bno', x, self.weights[ii]) + self.bias[ii]
                x = self.activation(x)
            x = torch.einsum('bni,nio->bno', x, self.weights[-1]) + self.bias[-1]
            return x

        elif len(x.shape) == 3:
            # for input shape (batch, net, input_feature), pass the input through their corresponding individual nets
            for ii in range(self.num_layers-1):
                x = torch.einsum('bni,nio->bno', x, self.weights[ii]) + self.bias[ii]
                x = self.activation(x)
            x = torch.einsum('bni,nio->bno', x, self.weights[-1]) + self.bias[-1]
            return x

        else:
            raise RuntimeError('Expect tensor of rank 2 or 3, but got {}'.format(len(input.shape)))


class FNN(nn.Module):
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dims_hidden_neurons: Tuple = (64, 64),
                 activation=torch.relu,
                 seed=0,
                 ):
        super(FNN, self).__init__()

        torch.manual_seed(seed)

        self.num_layers = len(dims_hidden_neurons) + 1
        self.activation = activation

        n_neurons = (dim_input,) + dims_hidden_neurons + (dim_output,)
        for ii, (dim_in, dim_out) in enumerate(zip(n_neurons[:-1], n_neurons[1:])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(ii))

    def forward(self, input: torch.Tensor):
        x = input
        for ii in range(self.num_layers-1):
            x = eval('self.activation(self.layer{}(x))'.format(ii))
        x = eval('self.layer{}(x)'.format(self.num_layers-1))
        return x

