from chapter import LayerBase
import numpy as np

######### 优化方法(Optimizer)见 method/optimizer #######


######## 参数初始化(Parameter Initialization) 见method/weight #####


######## BatchNorm1D #####
class BatchNorm1D(LayerBase):

    def __init__(self, momentum=0.9, epsilon=1e-5, optimizer=None):
        """
        参数说明：
        momentum：动量项，越趋于 1 表示对当前 Batch 的依赖程度越小，running_mean和running_var的计算越平滑
                    float型 (default: 0.9)

        epsilon：避免除数为0，float型 (default : 1e-5)
        optimizer：优化器
        """
        super().__init__(optimizer)

        self.n_in = None
        self.n_out = None
        self.epsilon = epsilon
        self.momentum = momentum
        self.params = {
            "scaler": None,
            "intercept": None,
            "running_var": None,
            "running_mean": None,
        }
        self.is_initialized = False

    def _init_params(self):
        scaler = np.random.rand(self.n_in)
        intercept = np.zeros(self.n_in)
        running_mean = np.zeros(self.n_in)
        running_var = np.ones(self.n_in)
        
        self.params = {
            "scaler": scaler,
            "intercept": intercept,
            "running_mean": running_mean,
            "running_var": running_var,
        }
        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }
        self.is_initialized = True

    def reset_running_stats(self):
        self.params["running_mean"] = np.zeros(self.n_in)
        self.params["running_var"] = np.ones(self.n_in)

    def forward(self, X, is_train=True, retain_derived=True):
        """
        Batch 训练时 BN 的前向传播，原理见上文。

        [train]: Y = scaler * norm(X) + intercept，其中 norm(X) = (X - mean(X)) / sqrt(var(X) + epsilon)

        [test]: Y = scaler * running_norm(X) + intercept，
                    其中 running_norm(X) = (X - running_mean) / sqrt(running_var + epsilon)
            
        参数说明：
        X：输入数组，为（n_samples, n_in），float型
        is_train：是否为训练阶段，bool型
        retain_derived：是否保留中间变量，以便反向传播时再次使用，bool型
        """
        if not self.is_initialized:
            self.n_in = self.n_out = X.shape[1]
            self._init_params()

        epsi, momentum = self.hyperparams["epsilon"], self.hyperparams["momentum"]
        rm, rv = self.params["running_mean"], self.params["running_var"]

        scaler, intercept = self.params["scaler"], self.params["intercept"]
        X_mean, X_var = self.params["running_mean"], self.params["running_var"]

        if is_train and retain_derived:
            X_mean, X_var = X.mean(axis=0), X.var(axis=0) 
            self.params["running_mean"] = momentum * rm + (1.0 - momentum) * X_mean
            self.params["running_var"] = momentum * rv + (1.0 - momentum) * X_var

        if retain_derived:
            self.X.append(X)

        X_hat = (X - X_mean) / np.sqrt(X_var + epsi)
        y = scaler * X_hat + intercept
        return y

    def backward(self, dLda, retain_grads=True):
        """
        BN 的反向传播，原理见上文。
        
        参数说明：
        dLda：关于损失的梯度，为（n_samples, n_out），float型
        retain_grads：是否计算中间变量的参数梯度，bool型
        """
        if not isinstance(dLda, list):
            dLda = [dLda]

        dX = []
        X = self.X
        for da, x in zip(dLda, X):
            dx, dScaler, dIntercept = self._bwd(da, x)
            dX.append(dx)

            if retain_grads:
                self.gradients["scaler"] += dScaler
                self.gradients["intercept"] += dIntercept

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLda, X):
        scaler = self.params["scaler"]
        epsi = self.hyperparams["epsilon"]

        n_ex, n_in = X.shape
        X_mean, X_var = X.mean(axis=0), X.var(axis=0)
        X_hat = (X - X_mean) / np.sqrt(X_var + epsi)
        
        dIntercept = dLda.sum(axis=0)
        dScaler = np.sum(dLda * X_hat, axis=0)
        dX_hat = dLda * scaler
        
        dX = (n_ex * dX_hat - dX_hat.sum(axis=0) - X_hat * (dX_hat * X_hat).sum(axis=0)) / (
            n_ex * np.sqrt(X_var + epsi)
        )

        return dX, dScaler, dIntercept
    
    @property
    def hyperparams(self):
        return {
            "layer": "BatchNorm1D",
            "acti_fn": None,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparams": self.optimizer.hyperparams,
            },
        }
