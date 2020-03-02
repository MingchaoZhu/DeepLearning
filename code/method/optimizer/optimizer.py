from abc import ABC, abstractmethod
import numpy as np
import re


class OptimizerBase(ABC):
    
    def __init__(self):
        pass
        
    def __call__(self, params, params_grad, params_name):
        """
        参数说明：
        params：待更新参数， 如权重矩阵 W；
        params_grad：待更新参数的梯度；
        params_name：待更新参数名；
        """
        return self.update(params, params_grad, params_name)
    
    @abstractmethod
    def update(self, params, params_grad, params_name):
        raise NotImplementedError

        
class SGD(OptimizerBase):
    """
    sgd 优化方法
    """
    
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr 
        self.cache = {}
        
    def __str__(self):
        return "SGD(lr={})".format(self.hyperparams["lr"])
    
    def update(self, params, params_grad, params_name):
        update_value = self.lr * params_grad
        return params - update_value
    
    @property
    def hyperparams(self):
        return {
            "op": "SGD",
            "lr": self.lr
        }

class Momentum(OptimizerBase):
    
    def __init__(
        self, lr=0.001, momentum=0.0, **kwargs
    ):
        """
        参数说明：
        lr： 学习率，float (default: 0.001)
        momentum：考虑 Momentum 时的 alpha，决定了之前的梯度贡献衰减得有多快，取值范围[0, 1]，默认0
        """
        super().__init__()
        self.lr = lr 
        self.momentum = momentum
        self.cache = {}

    def __str__(self):
        return "Momentum(lr={}, momentum={})".format(self.lr, self.momentum)

    def update(self, param, param_grad, param_name):
        C = self.cache
        lr, momentum = self.lr, self.momentum

        if param_name not in C:  # save v
            C[param_name] = np.zeros_like(param_grad)

        update = momentum * C[param_name] - lr * param_grad
        self.cache[param_name] = update
        return param + update
    
    @property
    def hyperparams(self):
        return {
            "op": "Momentum",
            "lr": self.lr,
            "momentum": self.momentum
        }
    

class AdaGrad(OptimizerBase):

    def __init__(self, lr=0.001, eps=1e-7, **kwargs):
        """
        参数说明：
        lr： 学习率，float (default: 0.001)
        eps：delta 项，防止分母为0
        """
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.cache = {}

    def __str__(self):
        return "AdaGrad(lr={}, eps={})".format(self.lr, self.eps)

    def update(self, param, param_grad, param_name):
        C = self.cache
        lr, eps = self.hyperparams["lr"], self.hyperparams["eps"]

        if param_name not in C:  # save r
            C[param_name] = np.zeros_like(param_grad)

        C[param_name] += param_grad ** 2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        return param - update

    @property
    def hyperparams(self):
        return {
            "op": "AdaGrad",
            "lr": self.lr,
            "eps": self.eps
        }
    
    
class RMSProp(OptimizerBase):
    
    def __init__(
        self, lr=0.001, decay=0.9, eps=1e-7, **kwargs
    ):
        """
        参数说明：
        lr： 学习率，float (default: 0.001)
        eps：delta 项，防止分母为0
        decay：衰减速率
        """
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.decay = decay
        self.cache = {}

    def __str__(self):
        return "RMSProp(lr={}, eps={}, decay={})".format(
            self.lr, self.eps, self.decay
        )

    def update(self, param, param_grad, param_name):
        C = self.cache
        lr, eps = self.hyperparams["lr"], self.hyperparams["eps"]
        decay = self.hyperparams["decay"]

        if param_name not in C:  # save r
            C[param_name] = np.zeros_like(param_grad)

        C[param_name] = decay * C[param_name] + (1 - decay) * param_grad ** 2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        return param - update
    
    @property
    def hyperparams(self):
        return {
            "op": "RMSProp",
            "lr": self.lr,
            "eps": self.eps,
            "decay": self.decay
        }    
    
    
class AdaDelta(OptimizerBase):
    
    def __init__(
        self, lr=0.001, decay=0.95, eps=1e-7, **kwargs
    ):
        """
        参数说明：
        lr： 学习率，float (default: 0.001)
        eps：delta 项，防止分母为0
        decay：衰减速率
        """
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.decay = decay
        self.cache = {}

    def __str__(self):
        return "AdaDelta(eps={}, decay={})".format(self.eps, self.decay)

    def update(self, param, param_grad, param_name):
        C = self.cache
        eps = self.hyperparams["eps"]
        decay = self.hyperparams["decay"]

        if param_name not in C:  # save r, delta_theta
            C[param_name] = {
                "r": np.zeros_like(param_grad),
                "d": np.zeros_like(param_grad)
            }

        C[param_name]["r"] = decay * C[param_name]["r"] + (1 - decay) * param_grad ** 2
        update = (np.sqrt(C[param_name]["d"] + eps)) * param_grad / (np.sqrt(C[param_name]["r"]) + eps)
        C[param_name]["d"] = decay * C[param_name]["d"] + (1 - decay) * update ** 2
        self.cache = C
        return param - update
    
    @property
    def hyperparams(self):
        return {
            "op": "AdaDelta",
            "eps": self.eps,
            "decay": self.decay
        }
    
    
class Adam(OptimizerBase):
    
    def __init__(
        self,
        lr=0.001,
        decay1=0.9,
        decay2=0.999,
        eps=1e-7,
        **kwargs
    ):
        """
        参数说明：
        lr： 学习率，float (default: 0.01)
        eps：delta 项，防止分母为0
        decay1：历史梯度的指数衰减速率，可以理解为考虑梯度均值 (default: 0.9)
        decay2：历史梯度平方的指数衰减速率，可以理解为考虑梯度方差 (default: 0.999)
        """
        super().__init__()
        self.lr = lr
        self.decay1 = decay1
        self.decay2 = decay2
        self.eps = eps
        self.cache = {}

    def __str__(self):
        return "Adam(lr={}, decay1={}, decay2={}, eps={})".format(
            self.lr, self.decay1, self.decay2, self.eps
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        C = self.cache
        d1, d2 = self.hyperparams["decay1"], self.hyperparams["decay2"]
        lr, eps= self.hyperparams["lr"], self.hyperparams["eps"]

        if param_name not in C:
            C[param_name] = {
                "t": 0,
                "mean": np.zeros_like(param_grad),
                "var": np.zeros_like(param_grad),
            }

        t = C[param_name]["t"] + 1
        mean = C[param_name]["mean"]
        var = C[param_name]["var"]

        C[param_name]["t"] = t
        C[param_name]["mean"] = d1 * mean + (1 - d1) * param_grad
        C[param_name]["var"] = d2 * var + (1 - d2) * param_grad ** 2
        self.cache = C

        m_hat = C[param_name]["mean"] / (1 - d1 ** t)
        v_hat = C[param_name]["var"] / (1 - d2 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return param - update

    @property
    def hyperparams(self):
        return {
            "op": "Adam",
            "lr": self.lr,
            "eps": self.eps,
            "decay1": self.decay1,
            "decay2": self.decay2
        }    
    
    
class OptimizerInitializer(ABC):
    
    def __init__(self, opti_name="sgd"):
        self.opti_name = opti_name
    
    def __call__(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        opti_str = self.opti_name.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, opti_str)])
        if "sgd" in opti_str:
            optimizer = SGD(**kwargs)
        elif "momentum" in opti_str:
            optimizer = Momentum(**kwargs)    
        elif "adagrad" in opti_str:
            optimizer = AdaGrad(**kwargs)
        elif "rmsprop" in opti_str:
            optimizer = RMSProp(**kwargs)
        elif "adadelta" in opti_str:
            optimizer = AdaDelta(**kwargs)
        elif "adam" in opti_str:
            optimizer = Adam(**kwargs)
        else:
            raise NotImplementedError("{}".format(opt_str))
        return optimizer
        
