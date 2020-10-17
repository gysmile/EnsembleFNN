import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms
from model import *

from tqdm import tqdm

# device = 'cpu'  # cpu
device = 'cuda:0'  # gpu

hidden_dims = (80, 64)
activation = torch.relu
lr = 0.001
num_nets = 8
seed = 0
batch_size = 128
training_iter = 4000
val_per_iter = 100

cuda = True if device == 'cuda:0' and torch.cuda.is_available() else False
Tensor = torch.cuda.DoubleTensor if cuda else torch.DoubleTensor
device = 'cuda' if cuda else 'cpu'
torch.set_default_tensor_type(Tensor)


training_set = MNIST('./data', train=True, download=True)
testing_set = MNIST('./data', train=False)
num_classes = 10

# preprocessing
X, Y = training_set.data, training_set.targets
X_test, Y_test = testing_set.data, testing_set.targets
X = X.view(X.shape[0], -1).double()
X_test = X_test.view(X_test.shape[0], -1).double()
X = (X - torch.mean(X, dim=1, keepdim=True))/torch.std(X, dim=1, keepdim=True)
X_test = (X_test - torch.mean(X_test, dim=1, keepdim=True))/torch.std(X_test, dim=1, keepdim=True)
X = X.to(device)
X_test = X_test.to(device)
Y = Y.to(device)
Y_test = Y_test.to(device)

criterion = nn.CrossEntropyLoss()  # takes pre-softmax logit for prediction, and integer index for target

ensemblefnn = EnsembleFNN(
    dim_input=training_set.data.shape[1] * training_set.data.shape[2],
    dim_output=num_classes,
    dims_hidden_neurons=hidden_dims,
    num_nets=num_nets,
    activation=activation,
    seed=seed,
)

fnn = FNN(
    dim_input=training_set.data.shape[1] * training_set.data.shape[2],
    dim_output=num_classes,
    dims_hidden_neurons=hidden_dims,
    activation=activation,
    seed=seed,
)

if cuda:
    ensemblefnn.cuda()
    fnn.cuda()
optimizer_fnn = torch.optim.Adam(fnn.parameters(), lr=lr)
optimizer_efnn = torch.optim.Adam(ensemblefnn.parameters(), lr=lr)


fnn_test_loss = []
for iter in tqdm(range(training_iter)):
    idx = np.random.choice(X.shape[0], batch_size)
    output = fnn(X[idx, :])
    loss = criterion(output, Y[idx])
    optimizer_fnn.zero_grad()
    loss.backward()
    optimizer_fnn.step()
    if iter % val_per_iter == 0:
        with torch.no_grad():
            output = fnn(X_test)
            loss = criterion(output, Y_test)
            fnn_test_loss.append(loss)


ensemblefnn_test_loss = []
ensemblefnn_test_loss_ind = [[] for ii in range(num_nets)]
for iter in tqdm(range(training_iter)):
    # idx = np.random.choice(X.shape[0], batch_size)  # all nets use sample mini batch
    idx = np.random.choice(X.shape[0], (batch_size, num_nets))  # different mini batch for different nets
    output = ensemblefnn(X[idx, :])
    # loss = criterion(torch.mean(output, dim=1), Y[idx])
    loss = criterion(output.contiguous().view(-1, num_classes), Y[idx].view(-1))
    optimizer_efnn.zero_grad()
    loss.backward()
    optimizer_efnn.step()
    if iter % val_per_iter == 0:
        with torch.no_grad():
            output = ensemblefnn(X_test)
            loss = criterion(torch.mean(output, dim=1), Y_test)
            ensemblefnn_test_loss.append(loss)
            for ii in range(num_nets):
                loss = criterion(output[:, ii, :], Y_test)
                ensemblefnn_test_loss_ind[ii].append(loss)


plt.plot(fnn_test_loss, label='fnn')
line = plt.plot(ensemblefnn_test_loss, label='ensemble fnn')
for ii in range(num_nets):
    plt.plot(ensemblefnn_test_loss_ind[ii], '--', color=line[0].get_color(), alpha=0.2)
plt.title("Testing loss")
plt.xlabel("Training iterations * {}".format(val_per_iter))
plt.legend()
plt.savefig('./figs/res.pdf', bbox_inches='tight')