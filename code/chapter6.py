from abc import ABC, abstractmethod
import numpy as np
import time
import re
import inspect
from collections import OrderedDict

import sys
sys.path.append('../')
from method.optimizer import OptimizerInitializer
from method.weight import WeightInitializer
from method.activation import ActivationInitializer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
    
    
class LayerBase(ABC):
    
    def __init__(self, optimizer="sgd"):
        self.X = []   # 网络层输入
        self.gradients = {}  # 网络层待梯度更新变量
        self.params = {}  # 网络层参数变量
        self.acti_fn = None   # 网络层激活函数
        self.optimizer = OptimizerInitializer(optimizer)()  # 网络层优化方法

    @abstractmethod
    def _init_params(self, **kwargs):
        """
        函数作用：初始化参数
        """
        raise NotImplementedError
        
    @abstractmethod
    def forward(self, X, **kwargs):  
        """
        函数作用：前向传播
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """
        函数作用：反向传播
        """
        raise NotImplementedError
        
    def flush_gradients(self):
        """
        函数作用：重置更新参数列表
        """
        self.X = []
        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)
            
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

    def update(self):
        """
        函数作用：更新参数
        """
        for k, v in self.gradients.items():
            if k in self.params:
                self.params[k] = self.optimizer(self.params[k], v, k)
    
    
class FullyConnected(LayerBase):
    """
    定义全连接层，实现 a=g(x*W+b)，前向传播输入x，返回a；反向传播输入
    """
    
    def __init__(self, n_out, acti_fn, init_w, optimizer=None):
        """
        参数说明：
        acti_fn：激活函数， str型
        init_w：权重初始化方法， str型
        n_out：隐藏层输出维数
        optimizer：优化方法
        """
        super().__init__(optimizer)
        
        self.n_in = None  # 隐藏层输入维数， int型
        self.n_out = n_out  # 隐藏层输出维数， int型
        self.acti_fn = ActivationInitializer(acti_fn)()
        self.init_w = init_w
        self.init_weights = WeightInitializer(mode=init_w)
        self.is_initialized = False  # 是否初始化， bool型变量
    
    def _init_params(self):
        b = np.zeros((1, self.n_out))
        W = self.init_weights((self.n_in, self.n_out))
        self.params = {"W": W, "b": b}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.derived_variables = {"Z": []}
        self.is_initialized = True
        
    def forward(self, X, retain_derived=True):
        """
        全连接网络的前向传播，原理见上文 反向传播算法 部分。
        
        参数说明：
        X：输入数组，为（n_samples, n_in），float型
        retain_derived：是否保留中间变量，以便反向传播时再次使用，bool型
        """
        if not self.is_initialized:  # 如果参数未初始化，先初始化参数
            self.n_in = X.shape[1]
            self._init_params()
            
        W = self.params["W"]
        b = self.params["b"]
        z = X @ W + b
        a = self.acti_fn.forward(z)
        
        if retain_derived:
            self.X.append(X)
            
        return a
    
    def backward(self, dLda, retain_grads=True):
        """
        全连接网络的反向传播，原理见上文 反向传播算法 部分。
        
        参数说明：
        dLda：关于损失的梯度，为（n_samples, n_out），float型
        retain_grads：是否计算中间变量的参数梯度，bool型
        """
        if not isinstance(dLda, list):
            dLda = [dLda]
        
        dX = []
        X = self.X
        for da, x in zip(dLda, X):
            dx, dw, db = self._bwd(da, x)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db
        
        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLda, X):
        W = self.params["W"]
        b = self.params["b"]

        Z = X @ W + b
        dZ = dLda * self.acti_fn.grad(Z)

        dX = dZ @ W.T
        dW = X.T @ dZ
        db = dZ.sum(axis=0, keepdims=True)
        return dX, dW, db
    
    @property
    def hyperparams(self):
        return {
            "layer": "FullyConnected",
            "init_w": self.init_w,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "acti_fn": str(self.acti_fn),
            "optimizer": {
                "hyperparams": self.optimizer.hyperparams,
            },
            "components": {
                k: v for k, v in self.params.items()
            }
        }
    
    
class ObjectiveBase(ABC):
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        """
        函数作用：计算损失
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        """
        函数作用：计算代价函数的梯度
        """
        raise NotImplementedError


class SquaredError(ObjectiveBase):
    """
    二次代价函数。
    """
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self):
        return "SquaredError"

    @staticmethod
    def loss(y_true, y_pred):
        """
        参数说明：
        y_true：训练的 n 个样本的真实值， 形状为(n,m)数组；
        y_pred：训练的 n 个样本的预测值， 形状为(n,m)数组；
        """
        (n, _) = y_true.shape
        return 0.5 * np.linalg.norm(y_pred - y_true) ** 2 / n

    @staticmethod
    def grad(y_true, y_pred, z, acti_fn):
        (n, _) = y_true.shape
        return (y_pred - y_true) * acti_fn.grad(z) / n


class CrossEntropy(ObjectiveBase):
    """
    交叉熵代价函数。
    """
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self):
        return "CrossEntropy"

    @staticmethod
    def loss(y_true, y_pred):
        """
        参数说明：
        y_true：训练的 n 个样本的真实值， 要求形状为(n,m)二进制（每个样本均为 one-hot 编码）；
        y_pred：训练的 n 个样本的预测值， 形状为(n,m)；
        """
        (n, _) = y_true.shape
        eps = np.finfo(float).eps  # 防止 np.log(0)
        cross_entropy = -np.sum(y_true * np.log(y_pred + eps)) / n 
        return cross_entropy
    
    @staticmethod
    def grad(y_true, y_pred):
        (n, _) = y_true.shape
        grad = (y_pred - y_true) / n
        return grad
    
    
def minibatch(X, batchsize=256, shuffle=True):
    """
    函数作用：将数据集分割成 batch， 基于 mini batch 训练。
    """
    N = X.shape[0]
    idx = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(idx)

    def mb_generator():
        for i in range(n_batches):
            yield idx[i * batchsize : (i + 1) * batchsize]

    return mb_generator(), n_batches


class DFN(object):
    
    def __init__(
        self,
        hidden_dims_1=None,
        hidden_dims_2=None,
        optimizer="sgd(lr=0.01)",
        init_w="std_normal",
        loss=CrossEntropy()
    ):
        self.optimizer = optimizer
        self.init_w = init_w
        self.loss = loss
        self.hidden_dims_1 = hidden_dims_1
        self.hidden_dims_2 = hidden_dims_2
        self.is_initialized = False
    
    def _set_params(self):
        """
        函数作用：模型初始化
        FC1 -> Sigmoid -> FC2 -> Softmax
        """
        self.layers = OrderedDict()
        self.layers["FC1"] = FullyConnected(
            n_out=self.hidden_dims_1,
            acti_fn="sigmoid", 
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.layers["FC2"] = FullyConnected(
            n_out=self.hidden_dims_2,
            acti_fn="affine(slope=1, intercept=0)",
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.is_initialized = True
    
    def forward(self, X_train):
        Xs = {}
        out = X_train
        for k, v in self.layers.items():
            Xs[k] = out
            out = v.forward(out)
        return out, Xs
    
    def backward(self, grad):
        dXs = {}
        out = grad
        for k, v in reversed(list(self.layers.items())):
            dXs[k] = out
            out = v.backward(out)
        return out, dXs
    
    def update(self):
        """
        函数作用：梯度更新
        """
        for k, v in reversed(list(self.layers.items())):
            v.update()
        self.flush_gradients()
    
    def flush_gradients(self, curr_loss=None):
        """
        函数作用：更新后重置梯度
        """
        for k, v in self.layers.items():
            v.flush_gradients()
    
    def fit(self, X_train, y_train, n_epochs=20, batch_size=64, verbose=False, epo_verbose=True):
        """
        参数说明：
        X_train：训练数据
        y_train：训练数据标签
        n_epochs：epoch 次数
        batch_size：每次 epoch 的 batch size
        verbose：是否每个 batch 输出损失
        epo_verbose：是否每个 epoch 输出损失
        """
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        if not self.is_initialized:
            self.n_features = X_train.shape[1]
            self._set_params()
        
        prev_loss = np.inf
        for i in range(n_epochs):
            loss, epoch_start = 0.0, time.time()
            batch_generator, n_batch = minibatch(X_train, self.batch_size, shuffle=True)

            for j, batch_idx in enumerate(batch_generator):
                batch_len, batch_start = len(batch_idx), time.time()
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                out, _ = self.forward(X_batch)
                y_pred_batch = softmax(out)
                batch_loss = self.loss(y_batch, y_pred_batch)
                grad = self.loss.grad(y_batch, y_pred_batch)
                _, _ = self.backward(grad)
                self.update()
                loss += batch_loss

                if self.verbose:
                    fstr = "\t[Batch {}/{}] Train loss: {:.3f} ({:.1f}s/batch)"
                    print(fstr.format(j + 1, n_batch, batch_loss, time.time() - batch_start))

            loss /= n_batch
            if epo_verbose:
                fstr = "[Epoch {}] Avg. loss: {:.3f}  Delta: {:.3f} ({:.2f}m/epoch)"
                print(fstr.format(i + 1, loss, prev_loss - loss, (time.time() - epoch_start) / 60.0))
            prev_loss = loss
            
    def evaluate(self, X_test, y_test, batch_size=128):
        acc = 0.0
        batch_generator, n_batch = minibatch(X_test, batch_size, shuffle=True)
        for j, batch_idx in enumerate(batch_generator):
            batch_len, batch_start = len(batch_idx), time.time()
            X_batch, y_batch = X_test[batch_idx], y_test[batch_idx]
            y_pred_batch, _ = self.forward(X_batch)
            y_pred_batch = np.argmax(y_pred_batch, axis=1)
            y_batch = np.argmax(y_batch, axis=1)
            acc += np.sum(y_pred_batch == y_batch)
        return acc / X_test.shape[0]
    
    @property
    def hyperparams(self):
        return {
            "init_w": self.init_w,
            "loss": str(self.loss),
            "optimizer": self.optimizer,
            "hidden_dims_1": self.hidden_dims_1,
            "hidden_dims_2": self.hidden_dims_2,
            "components": {k: v.params for k, v in self.layers.items()}
        }
    