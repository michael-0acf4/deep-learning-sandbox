import torch
import matplotlib.pyplot as plt

# This small lib will be used throughout the rest of the project
from scratch.lib import (
    Sigmoid,
    ReLU,
    MSE as lossMSE,
    SGD as optSGD,
    Adam as optAdam,
    MLP as SeqLinear,
    LinearLayer,
)


xs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
ys = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

torch.manual_seed(177013)

net_sgd = SeqLinear(
    nout=1,
    layers=[LinearLayer(2, 4), ReLU(), LinearLayer(4, 1), Sigmoid()],
    criterion=lossMSE(),
    optimizer=optSGD(0.1),  # !
)
net_sgd.train(xs, ys, 2000)

net_adam = SeqLinear(
    nout=1,
    layers=[LinearLayer(2, 4), ReLU(), LinearLayer(4, 1), Sigmoid()],
    criterion=lossMSE(),
    optimizer=optAdam(0.1),  # !
)
net_adam.train(xs, ys, 2000)

plt.plot(net_sgd.losses)
plt.plot(net_adam.losses)
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.show()
