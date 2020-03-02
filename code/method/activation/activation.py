from abc import ABC, abstractmethod
import numpy as np
import re


class ActivationBase(ABC):
    
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.forward(z)

    @abstractmethod
    def forward(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        raise NotImplementedError


class Sigmoid(ActivationBase):
    """
    Sigmoid(x) = 1 / (1 + e^(-x))
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        return self.forward(x) * (1 - self.forward(x))


class Tanh(ActivationBase):
    """
    Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def forward(self, z):
        return np.tanh(z)

    def grad(self, x):
        return 1 - np.tanh(x) ** 2
    
    
class ReLU(ActivationBase):
    """
    ReLU(x) =
            x   if x > 0
            0   otherwise
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def forward(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)


class LeakyReLU(ActivationBase):
    """
    LeakyReLU(x) =
            alpha * x   if x < 0
            x           otherwise
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def forward(self, z):
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def grad(self, x):
        out = np.ones_like(x)
        out[x < 0] *= self.alpha
        return out


class Affine(ActivationBase):
    """
    Affine(x) = slope * x + intercept
    """

    def __init__(self, slope=1, intercept=0):
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def forward(self, z):
        return self.slope * z + self.intercept

    def grad(self, x):
        return self.slope * np.ones_like(x)


class SoftPlus(ActivationBase):
    """
    SoftPlus(x) = log(1 + e^x)
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SoftPlus"

    def forward(self, z):
        return np.log(np.exp(z) + 1)

    def grad(self, x):
        return np.exp(x) / (np.exp(x) + 1)
    
    
class ELU(ActivationBase):
    """
    ELU(x) =
            x                   if x >= 0
            alpha * (e^x - 1)   otherwise
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "ELU(alpha={})".format(self.alpha)

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def grad(self, x):
        return np.where(x >= 0, np.ones_like(x), self.alpha * np.exp(x))


class Exponential(ActivationBase):
    """
    Exponential(x) = e^x
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Exponential"

    def forward(self, z):
        return np.exp(z)

    def grad(self, x):
        return np.exp(x)


class SELU(ActivationBase):
    """
    SELU(x) = scale * ELU(x, alpha)
            = scale * x                     if x >= 0
              scale * [alpha * (e^x - 1)]   otherwise
    """

    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.elu = ELU(alpha=self.alpha)
        super().__init__()

    def __str__(self):
        return "SELU"

    def forward(self, z):
        return self.scale * self.elu.forward(z)

    def grad(self, x):
        return np.where(
            x >= 0, np.ones_like(x) * self.scale, np.exp(x) * self.alpha * self.scale
        )


class HardSigmoid(ActivationBase):
    """
    HardSigmoid(x) =
            0               if x < -2.5
            0.2 * x + 0.5   if -2.5 <= x <= 2.5.
            1               if x > 2.5
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Hard Sigmoid"

    def forward(self, z):
        return np.clip((0.2 * z) + 0.5, 0.0, 1.0)

    def grad(self, x):
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    
class ActivationInitializer(object):
    
    def __init__(self, acti_name="affine(slope=1, intercept=0)"):
        self.acti_name = acti_name

    def __call__(self):
        acti_str = self.acti_name.lower()
        if acti_str == "relu":
            acti_fn = ReLU()
        elif acti_str == "tanh":
            acti_fn = Tanh()
        elif acti_str == "sigmoid":
            acti_fn = Sigmoid()
        elif "affine" in acti_str:
            r = r"affine\(slope=(.*), intercept=(.*)\)"
            slope, intercept = re.match(r, acti_str).groups()
            acti_fn = Affine(float(slope), float(intercept))
        elif "leaky relu" in acti_str:
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, acti_str).groups()[0]
            acti_fn = LeakyReLU(float(alpha))
        else:
            raise ValueError("Unknown activation: {}".format(acti_str))
        return acti_fn
