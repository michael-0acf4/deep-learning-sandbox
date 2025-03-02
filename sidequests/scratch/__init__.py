"""
Yet another Deep Learning Library (YaDLL)
"""

from typing import List
import torch
import torch.nn.functional as F

import math


# Activation functions
class Activation:
    pass


class ReLU(Activation):
    def __call__(self, val: torch.Tensor):
        return val.relu()


class Sigmoid(Activation):
    def __call__(self, val: torch.Tensor):
        return val.sigmoid()


class Tanh(Activation):
    def __call__(self, val: torch.Tensor):
        return val.tanh()


# Loss functions
class LossCriterion:
    def __call__(self, hs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        raise Exception("unimplemented")


class MSE(LossCriterion):
    def __call__(self, hs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        return torch.sum((hs - ys) ** 2) / hs.numel()

# Optimizers
class Optimizer:
    def __call__(self, params: List[torch.Tensor]):
        raise Exception("unimplemented")


class SGD(Optimizer):
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, params: List[torch.Tensor]):
        for param in params:
            param.data -= self.lr * param.grad


class Adam(Optimizer):
    def __init__(
        self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

        # AdamW only
        self.weight_decay = None

    def __call__(self, params: List[torch.Tensor]):
        # Note: t should increase everytime we optimize through Adam
        # and does not depend on the batching strategy
        self.t += 1

        for param in params:
            if param not in self.m:
                self.m[param] = torch.zeros_like(param)
                self.v[param] = torch.zeros_like(param)

            grad = param.grad

            if grad is not None:
                # Weight decay (AdamW specific)
                # Note: WEIGHT DECAY IS NOT L2 REGULARIZATION (https://arxiv.org/abs/1711.05101)
                #                    ^^^^^^

                # Weight decay is defined in the paper as
                # w <- (1 - d) * w - k*∇L so roughly.. wi <- (1 - d) * wi - k*dL/dwi which is very intuitive
                # and after permutting the terms, then cleaning up the scalars... it is also equivalent to
                # w <- w - n*(∇L + m*wi)
                # but in our case, we decouple the decay step from the update
                # w <- w - n*m*w
                # w <- w - n*∇L (for AdamW replace ∇L by the boosted Adam grad)

                # Ln regularizations are terms added DIRECTLY onto the loss so that it stays within the boundaries of whatever vector norm-ish it refers to.
                # The consensus confusion happens because L2 reg is mathematically equivalent (by a scalar bit*) to weight decay in standard SGD!
                #
                # L2 norm by def is euclidean norm: L <- L + k* Σ w_i²
                # then after doing the math, it basically folds into something like (∇L + m*wi)
                # for AdamW it is not that! it is (adamthingy + m*wi) + decoupling of the learned lr!
                # (notice the lr)

                if self.weight_decay is not None:
                    # Weight decay just so happens to work fairly well with Adam (now AdamW)
                    param.data -= self.lr * self.weight_decay * param.data

                # Biased first and second moment estimates
                self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
                self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (
                    grad**2
                )

                # Bias-corrected moment estimates
                m_hat = self.m[param] / (1 - self.beta1**self.t)
                v_hat = self.v[param] / (1 - self.beta2**self.t)

                param.data += -1 * self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


# TODO: BatchNorm
# TODO: LayerNorm


class AdamW(Adam):
    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay


class LinearLayer:
    def __init__(self, inp: int, out: int):
        gain = math.sqrt(2 / (inp + out))

        self.w = torch.empty(out, inp, requires_grad=True, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(self.w, gain)

        self.b = torch.rand(out, 1, requires_grad=True, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(self.w, gain)

    def forward(self, x):
        # (N, I) . (I, 1) + (N, 1) = (N, 1)
        return self.w @ x + self.b


class MLP:
    def __init__(
        self,
        nout: int,
        criterion: LossCriterion,
        optimizer: Optimizer,
        layers: List[LinearLayer],
    ):
        self.nout = nout
        self.criterion = criterion
        self.optimizer = optimizer
        self.layers = layers
        self.losses = []

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                x = layer.forward(x)
            elif isinstance(layer, Activation):
                x = layer(x)
            else:
                raise ValueError("Unknown layer type")
        if x.numel() != self.nout:
            raise ValueError(
                f"MLP expects {self.nout} outputs, got {x.numel()} instead"
            )
        return x

    def train(self, xs: torch.Tensor, ys: torch.Tensor, epoch=100):
        for i in range(epoch):
            self.reset()

            outputs = torch.cat([self.forward(x.view(len(x), 1)) for x in xs])
            loss = self.criterion(outputs, ys)

            self.losses.append(loss.item())

            loss.backward()
            with torch.no_grad():
                self.optimizer(self.params())

        print(f"{i+1}.. Loss {loss.item()}")

    def params(self) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                params += [layer.w, layer.b]
        return params

    def reset(self):
        for p in self.params():
            p.grad = None


class Conv2D:
    def __init__(
        self,
        n_chan: int,  # input channels (e.g. for RGB that would be 3)
        n_ker: int,  # output channels, each channel encodes a kernel that can exrtract a feature
        ker_dim: int,
        stride: int | torch.SymInt = 1,  # offset
        padding: (
            int | str
        ) = "same",  # padding is computed so that we keep the original dim
    ):
        # (K, C, D, D)
        self.kernels = torch.rand(
            (n_ker, n_chan, ker_dim, ker_dim), requires_grad=True, dtype=torch.float32
        )
        self.bias = torch.rand(n_ker, requires_grad=True)  # One bias per kernel

        self.stride = stride  # kernel stride this amount pixel at a time (i.e. skip)
        self.padding = padding  # pad original image

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        # x_batch -> (N, C, W, H)
        # conv    -> (K, C, D, D)
        # out     -> (N, K, 1 + (W + 2P - D) / S, 1 + (H + 2P - D) / S)
        assert x_batch.ndimension() == 4
        # equiv. return conv(x) + bias
        return F.conv2d(x_batch, self.kernels, self.bias, self.stride, self.padding)


class DownSampler2D:
    def __init__(
        self,
        ker_dim: int,  # will encode a square window that will glide through the input
        stride: int | torch.SymInt = 1,
        padding: int | None = None,
    ):
        self.kernel_dim = ker_dim
        self.stride = stride
        self.padding = padding

    def forward(self, conv: torch.Tensor) -> torch.Tensor:
        """
        `conv: (N_batch, K_out_chan, W', H')`
        """
        raise Exception("unimplemented")


class MaxPool2D(DownSampler2D):
    def __init__(
        self,
        ker_dim: int,
        stride: int | torch.SymInt = 1,
        padding: int | None = None,
    ):
        super().__init__(ker_dim, stride, padding)

    def forward(self, conv: torch.Tensor) -> torch.Tensor:
        assert conv.ndimension() == 4
        return F.max_pool2d(conv, self.kernel_dim, self.stride, self.padding)


class AvgPool2D(DownSampler2D):
    def __init__(
        self,
        ker_dim: int,
        stride: int | torch.SymInt = 1,
        padding: int | None = None,
    ):
        super().__init__(ker_dim, stride, padding)

    def forward(self, conv: torch.Tensor) -> torch.Tensor:
        assert conv.ndimension() == 4
        return F.avg_pool2d(conv, self.kernel_dim, self.stride, self.padding)
