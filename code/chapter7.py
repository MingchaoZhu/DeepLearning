from abc import ABC, abstractmethod
import numpy as np
import re


#########---Regularizer---######
class RegularizerBase(ABC):
    
    def __init__(self, **kwargs):
        super().__init__()
    
    @abstractmethod
    def loss(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def grad(self, **kwargs):
        raise NotImplementedError

class L1Regularizer(RegularizerBase):
    
    def __init__(self, lambd=0.001):
        super().__init__()
        self.lambd = lambd
    
    def loss(self, params):
        loss = 0
        pattern = re.compile(r'^W\d+')
        for key, val in params.items():
            if pattern.match(key):
                loss +=  0.5 * np.sum(np.abs(val)) * self.lambd
        return loss
    
    def grad(self, params):
        for key, val in params.items():
            grad = self.lambd * np.sign(val)
        return grad
    
class L2Regularizer(RegularizerBase):
    
    def __init__(self, lambd=0.001):
        super().__init__()
        self.lambd = lambd
        
    def loss(self, params):
        loss = 0
        for key, val in params.items():
            loss +=  0.5 * np.sum(np.square(val)) * self.lambd
        return loss
    
    def grad(self, params):
        for key, val in params.items():
            grad = self.lambd * val
        return grad
    
class RegularizerInitializer(object):
    
    def __init__(self, regular_name="l2"):
        self.regular_name = regular_name
    
    def __call__(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        regular_str = self.regular_name.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, regular_str)])
        if  "l1" in regular_str.lower():
            regular = L1Regularizer(**kwargs)
        elif "l2" in regular_str.lower():
            regular = L2Regularizer(**kwargs)
        else:
            raise ValueError("Unrecognized regular: {}".format(regular_str))
        return regular
    

#######----Dataset Augmentation----####
class Image(object):
    
    def __init__(self, image):
        self._set_params(image)
        
    def _set_params(self, image):
        self.img = image 
        self.row = image.shape[0] # 图像高度
        self.col = image.shape[1] # 图像宽度
        self.transform = None

    def Translation(self, delta_x, delta_y):
        """
        平移。
        
        参数说明：
        delta_x：控制左右平移，若大于0左移，小于0右移
        delta_y：控制上下平移，若大于0上移，小于0下移
        """
        self.transform = np.array([[1, 0, delta_x], 
                                   [0, 1, delta_y], 
                                   [0,  0,  1]])

    def Resize(self, alpha):
        """
        缩放。
        
        参数说明：
        alpha：缩放因子，不进行缩放设置为1
        """
        self.transform = np.array([[alpha, 0, 0], 
                                   [0, alpha, 0], 
                                   [0,  0,  1]])

    def HorMirror(self): 
        """
        水平镜像。
        """
        self.transform = np.array([[1,  0,  0], 
                                   [0, -1, self.col-1], 
                                   [0,  0,  1]])

    def VerMirror(self): 
        """
        垂直镜像。
        """
        self.transform = np.array([[-1, 0, self.row-1], 
                                   [0,  1,  0], 
                                   [0,  0,  1]])

    def Rotate(self, angle): 
        """
        旋转。
        
        参数说明：
        angle：旋转角度
        """
        self.transform = np.array([[math.cos(angle),-math.sin(angle),0],
                                   [math.sin(angle), math.cos(angle),0],
                                   [    0,              0,         1]])        

    def operate(self):
        temp = np.zeros(self.img.shape, dtype=self.img.dtype)
        for i in range(self.row):
            for j in range(self.col):
                temp_pos = np.array([i, j, 1])
                [x,y,z] = np.dot(self.transform, temp_pos)
                x = int(x)
                y = int(y)

                if x>=self.row or y>=self.col or x<0 or y<0:
                    temp[i,j,:] = 0
                else:
                    temp[i,j,:] = self.img[x,y]
        return temp
    
    def __call__(self, act):
        r = r"([a-zA-Z]*)=([^,)]*)"
        act_str = act.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, act_str)])
        if "translation" in act_str:
            self.Translation(**kwargs)
        elif "resize" in act_str:
            self.Resize(**kwargs)
        elif "hormirror" in act_str:
            self.HorMirror(**kwargs)
        elif "vermirror" in act_str:
            self.VerMirror(**kwargs)
        elif "rotate" in act_str:
            self.Rotate(**kwargs)
        return self.operate()

    
#######----Early Stopping----####
def early_stopping(valid):
    """
    参数说明：
    valid：验证集正确率列表
    """
    if len(valid) > 5:
        if valid[-1] < valid[-5] and valid[-2] < valid[-5] and valid[-3] < valid[-5] and valid[-4] < valid[-5]:
            return True
    return False


#####---Bagging--#####
def bootstrap_sample(X, Y):
    N, M = X.shape
    idxs = np.random.choice(N, N, replace=True)
    return X[idxs], Y[idxs]

class BaggingModel(object):

    def __init__(self, n_models):
        """
        参数说明：
        n_models：网络模型数目
        """
        self.models = []
        self.n_models = n_models

    def fit(self, X, Y):
        self.models = []
        for i in range(self.n_models):
            print("training {} base model:".format(i))
            X_samp, Y_samp = bootstrap_sample(X, Y)
            model = DFN(hidden_dims_1=200, hidden_dims_2=10)
            model.fit(X_samp, Y_samp)
            self.models.append(model)

    def predict(self, X):
        model_preds = np.array([[np.argmax(t.forward(x)[0]) for x in X] for t in self.models])
        return self._vote(model_preds)

    def _vote(self, predictions):
        out = [np.bincount(x).argmax() for x in predictions.T]
        return np.array(out)
    
    def evaluate(self, X_test, y_test):
        acc = 0.0
        y_pred = self.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        acc += np.sum(y_pred == y_true)
        return acc / X_test.shape[0]
    

#####----Dropout----#######
class Dropout(ABC):
    
    def __init__(self, wrapped_layer, p):
        """
        参数说明：
        wrapped_layer：被 dropout 的层
        p：神经元保留率
        """
        super().__init__()
        self._base_layer = wrapped_layer
        self.p = p
        self._init_wrapper_params()
        
    def _init_wrapper_params(self):
        self._wrapper_derived_variables = {"dropout_mask": None}
        self._wrapper_hyperparams = {"wrapper": "Dropout", "p": self.p}
        
    def flush_gradients(self):
        """
        函数作用：调用 base layer 重置更新参数列表
        """
        self._base_layer.flush_gradients()
        
    def update(self):
        """
        函数作用：调用 base layer 更新参数
        """
        self._base_layer.update()
        
    def forward(self, X, is_train=True):
        """
        参数说明：
        X：输入数组；
        is_train：是否为训练阶段，bool型；
        """
        mask = np.ones(X.shape).astype(bool)
        if is_train:
            mask = (np.random.rand(*X.shape) < self.p) / self.p
            X = mask * X
        self._wrapper_derived_variables["dropout_mask"] = mask
        return self._base_layer.forward(X)
        
    def backward(self, dLda):
        return self._base_layer.backward(dLda)
    
    @property
    def hyperparams(self):
        hp = self._base_layer.hyperparams
        hpw = self._wrapper_hyperparams
        if "wrappers" in hp:
            hp["wrappers"].append(hpw)
        else:
            hp["wrappers"] = [hpw]
        return hp
