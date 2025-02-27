"""
Yet another Deep Learning Library (YaDLL)
"""

from typing import List
import torch
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
                if self.weight_decay is not None:
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
