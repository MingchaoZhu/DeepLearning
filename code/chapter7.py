from abc import ABC, abstractmethod
import numpy as np
import math
import re
import progressbar
from chapter5 import RegressionTree, DecisionTree, ClassificationTree

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


#####----Bagging----#######
# 进度条
bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

def get_random_subsets(X, y, n_subsets, replacements=True):
    """从训练数据中抽取数据子集 (默认可重复抽样)"""
    n_samples = np.shape(X)[0]
    # 将 X 和 y 拼接，并将元素随机排序
    Xy = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(Xy)
    subsets = []
    # 如果抽样时不重复抽样，可以只使用 50% 的训练数据；如果抽样时可重复抽样，使用全部的训练数据，默认可重复抽样
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples      
    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements)
        X = Xy[idx][:, :-1]
        y = Xy[idx][:, -1]
        subsets.append([X, y])
    return subsets


class Bagging():
    """
    Bagging分类器。使用一组分类树，这些分类树使用特征训练数据的随机子集。
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators    # 树的数目
        self.min_samples_split = min_samples_split   # 分割所需的最小样本数
        self.min_gain = min_gain            # 分割所需的最小纯度 (最小信息增益)
        self.max_depth = max_depth          # 树的最大深度
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        # 初始化决策树
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth))

    def fit(self, X, y):
        # 对每棵树选择数据集的随机子集
        subsets = get_random_subsets(X, y, self.n_estimators)
        for i in self.progressbar(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            # 用特征子集和真实值训练一棵子模型 (这里的数据也是训练数据集的随机子集)
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # 每棵决策树都在数据上预测
        for i, tree in enumerate(self.trees):
            # 基于特征做出预测
            prediction = tree.predict(X)
            y_preds[:, i] = prediction
            
        y_pred = []
        # 对每个样本，选择最常见的类别作为预测
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

    
#####----RandomForest----#######
class RandomForest():
    """
    随机森林分类器。使用一组分类树，这些分类树使用特征的随机子集训练数据的随机子集。
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators    # 树的数目
        self.max_features = max_features    # 每棵树的最大使用特征数
        self.min_samples_split = min_samples_split   # 分割所需的最小样本数
        self.min_gain = min_gain            # 分割所需的最小纯度 (最小信息增益)
        self.max_depth = max_depth          # 树的最大深度
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        # 初始化决策树
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth))

    def fit(self, X, y):
        n_features = np.shape(X)[1]
        # 如果 max_features 没有定义，取默认值 sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # 对每棵树选择数据集的随机子集
        subsets = get_random_subsets(X, y, self.n_estimators)

        for i in self.progressbar(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            # 选择特征的随机子集
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # 保存特征的索引用于预测
            self.trees[i].feature_indices = idx
            # 选择索引对应的特征
            X_subset = X_subset[:, idx]
            # 用特征子集和真实值训练一棵子模型 (这里的数据也是训练数据集的随机子集)
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # 每棵决策树都在数据上预测
        for i, tree in enumerate(self.trees):
            # 使用该决策树训练使用的特征
            idx = tree.feature_indices
            # 基于特征做出预测
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
            
        y_pred = []
        # 对每个样本，选择最常见的类别作为预测
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

    
#####----Adaboost----#######
# 决策树桩，作为 Adaboost 算法的弱分类器 (基分类器)
class DecisionStump():
    
    def __init__(self):
        self.polarity = 1            # 表示决策树桩默认输出的类别为 1 或是 -1
        self.feature_index = None    # 用于分类的特征索引
        self.threshold = None        # 特征的阈值
        self.alpha = None            # 表示分类器准确性的值

class Adaboost():
    """
    Adaboost 算法。
    """
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators    # 将使用的弱分类器的数量
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        # 初始化权重 (上文中的 D)，均为 1/N
        w = np.full(n_samples, (1 / n_samples))
        self.trees = []
        # 迭代过程
        for _ in self.progressbar(range(self.n_estimators)):
            tree = DecisionStump()
            min_error = float('inf')    # 使用某一特征值的阈值预测样本的最小误差
            # 迭代遍历每个 (不重复的) 特征值，查找预测 y 的最佳阈值
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # 将该特征的每个特征值作为阈值
                for threshold in unique_values:
                    p = 1
                    # 将所有样本预测默认值可以设置为 1
                    prediction = np.ones(np.shape(y))
                    # 低于特征值阈值的预测改为 -1
                    prediction[X[:, feature_i] < threshold] = -1
                    # 计算错误率
                    error = sum(w[y != prediction])
                    # 如果错误率超过 50%，我们反转决策树桩默认输出的类别
                    # 比如 error = 0.8 => (1 - error) = 0.2，
                    # 原来计算的是输出到类别 1 的概率，类别 1 作为默认类别。反转后类别 0 作为默认类别
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    # 如果这个阈值导致最小的错误率，则保存
                    if error < min_error:
                        tree.polarity = p
                        tree.threshold = threshold
                        tree.feature_index = feature_i
                        min_error = error
                        
            # 计算用于更新样本权值的 alpha 值，也是作为基分类器的系数。
            tree.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # 将所有样本预测默认值设置为 1
            predictions = np.ones(np.shape(y))
            # 如果特征值低于阈值，则修改预测结果，这里还需要考虑弱分类器的默认输出类别
            negative_idx = (tree.polarity * X[:, tree.feature_index] < tree.polarity * tree.threshold)
            predictions[negative_idx] = -1
            # 计算新权值，未正确分类样本的权值增大，正确分类样本的权值减小
            w *= np.exp(-tree.alpha * y * predictions)
            w /= np.sum(w)
            # 保存分类器
            self.trees.append(tree)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # 用每一个基分类器预测样本
        for tree in self.trees:
            # 将所有样本预测默认值设置为 1
            predictions = np.ones(np.shape(y_pred))
            negative_idx = (tree.polarity * X[:, tree.feature_index] < tree.polarity * tree.threshold)
            predictions[negative_idx] = -1
            # 对基分类器加权求和，权重 alpha
            y_pred += tree.alpha * predictions
        # 返回预测结果 1 或 -1
        y_pred = np.sign(y_pred).flatten()
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

    
#####----GBDT----#######
class Loss(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod    
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    @abstractmethod    
    def grad(self, y, y_pred):
        raise NotImplementedError()

class SquareLoss(Loss):
    
    def __init__(self): 
        pass

    def loss(self, y, y_pred):
        pass

    def grad(self, y, y_pred):
        return -(y - y_pred)
    
    def hess(self, y, y_pred):
        return 1

class CrossEntropyLoss(Loss):
    
    def __init__(self): 
        pass

    def loss(self, y, y_pred):
        pass

    def grad(self, y, y_pred):
        return - (y - y_pred)  
    
    def hess(self, y, y_pred):
        return y_pred * (1-y_pred)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def line_search(self, y, y_pred, h_pred):
    Lp = 2 * np.sum((y - y_pred) * h_pred)
    Lpp = np.sum(h_pred * h_pred)
    return 1 if np.sum(Lpp) == 0 else Lp / Lpp


def to_categorical(x, n_classes=None):
    """
    One-hot编码
    """
    if not n_classes:
        n_classes = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_classes))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


class GradientBoostingDecisionTree(object):
    """
    GBDT 算法。用一组基学习器 (回归树) 学习损失函数的梯度。
    """
    def __init__(self, n_estimators, learning_rate=1, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), is_regression=False, line_search=False):
        self.n_estimators = n_estimators         # 迭代的次数
        self.learning_rate = learning_rate       # 训练过程中沿着负梯度走的步长，也就是学习率
        self.min_samples_split = min_samples_split    # 分割所需的最小样本数
        self.min_impurity = min_impurity         # 分割所需的最小纯度
        self.max_depth = max_depth               # 树的最大深度
        self.is_regression = is_regression       # 分类问题或回归问题
        self.line_search = line_search           # 是否使用 line search
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)        
        # 回归问题采用基础的平方损失，分类问题采用交叉熵损失
        self.loss = SquareLoss()
        if not self.is_regression:
            self.loss = CrossEntropyLoss()

    def fit(self, X, Y):
        # 分类问题将 Y 转化为 one-hot 编码
        if not self.is_regression:
            Y = to_categorical(Y.flatten())
        else:
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
        self.out_dims = Y.shape[1]
        self.trees = np.empty((self.n_estimators, self.out_dims), dtype=object)
        Y_pred = np.full(np.shape(Y), np.mean(Y, axis=0))
        self.weights = np.ones((self.n_estimators, self.out_dims))
        self.weights[1:, :] *= self.learning_rate
        # 迭代过程
        for i in self.progressbar(range(self.n_estimators)):
            for c in range(self.out_dims):
                tree = RegressionTree(
                        min_samples_split=self.min_samples_split,
                        min_impurity=self.min_impurity,
                        max_depth=self.max_depth)
                # 计算损失的梯度，并用梯度进行训练
                if not self.is_regression:   
                    Y_hat = softmax(Y_pred)
                    y, y_pred = Y[:, c], Y_hat[:, c]
                else:
                    y, y_pred = Y[:, c], Y_pred[:, c]
                neg_grad = -1 * self.loss.grad(y, y_pred)
                tree.fit(X, neg_grad)
                # 用新的基学习器进行预测
                h_pred = tree.predict(X)
                # line search
                if self.line_search == True:
                    self.weights[i, c] *= line_search(y, y_pred, h_pred)
                # 加法模型中添加基学习器的预测，得到最新迭代下的加法模型预测
                Y_pred[:, c] += np.multiply(self.weights[i, c], h_pred)
                self.trees[i, c] = tree
    
    def predict(self, X):
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        # 生成预测
        for c in range(self.out_dims):
            y_pred = np.array([])
            for i in range(self.n_estimators):
                update = np.multiply(self.weights[i, c], self.trees[i, c].predict(X))
                y_pred = update if not y_pred.any() else y_pred + update
            Y_pred[:, c] = y_pred
        if not self.is_regression: 
            # 分类问题输出最可能类别
            Y_pred = Y_pred.argmax(axis=1)
        return Y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy


class GradientBoostingRegressor(GradientBoostingDecisionTree):
    
    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), is_regression=True, line_search=False):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_impurity,
            max_depth=max_depth,
            is_regression=is_regression,
            line_search=line_search)


class GradientBoostingClassifier(GradientBoostingDecisionTree):
    
    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), is_regression=False, line_search=False):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_impurity,
            max_depth=max_depth,
            is_regression=is_regression,
            line_search=line_search)

        
#####----XGBoost----#######
class XGBoostRegressionTree(DecisionTree):
    """
    XGBoost 回归树。此处基于第五章介绍的决策树，故采用贪心算法找到特征上分裂点 (枚举特征上所有可能的分裂点)。
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None, gamma=0., lambd=0.):
        super(XGBoostRegressionTree, self).__init__(min_impurity=min_impurity, 
            min_samples_split=min_samples_split, 
            max_depth=max_depth)
        self.gamma = gamma   # 叶子节点的数目的惩罚系数
        self.lambd = lambd   # 叶子节点的权重的惩罚系数
        self.loss = loss     # 损失函数
    
    def _split(self, y):
        # y 包含 y_true 在左半列，y_pred 在右半列
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        # 计算信息
        nominator = np.power((y * self.loss.grad(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return nominator / (denominator + self.lambd)

    def _gain_by_taylor(self, y, y1, y2):
        # 分割为左子树和右子树
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        # 计算信息增益
        return 0.5 * (true_gain + false_gain - gain) - self.gamma

    def _approximate_update(self, y):
        y, y_pred = self._split(y)
        # 计算叶节点权重
        gradient = self.loss.grad(y, y_pred).sum()
        hessian = self.loss.hess(y, y_pred).sum()
        leaf_approximation = -gradient / (hessian + self.lambd)
        return leaf_approximation

    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)


class XGBoost(object):
    """
    XGBoost学习器。
    """
    def __init__(self, n_estimators=200, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, is_regression=False, gamma=0., lambd=0.):
        self.n_estimators = n_estimators            # 树的数目
        self.learning_rate = learning_rate          # 训练过程中沿着负梯度走的步长，也就是学习率
        self.min_samples_split = min_samples_split  # 分割所需的最小样本数
        self.min_impurity = min_impurity            # 分割所需的最小纯度
        self.max_depth = max_depth                  # 树的最大深度
        self.gamma = gamma                          # 叶子节点的数目的惩罚系数
        self.lambd = lambd                          # 叶子节点的权重的惩罚系数
        self.is_regression = is_regression          # 分类或回归问题
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        # 回归问题采用基础的平方损失，分类问题采用交叉熵损失
        self.loss = SquareLoss()
        if not self.is_regression:
            self.loss = CrossEntropyLoss()

    def fit(self, X, Y):
        # 分类问题将 Y 转化为 one-hot 编码
        if not self.is_regression:
            Y = to_categorical(Y.flatten())
        else:
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
        self.out_dims = Y.shape[1]
        self.trees = np.empty((self.n_estimators, self.out_dims), dtype=object)
        Y_pred = np.zeros(np.shape(Y))
        self.weights = np.ones((self.n_estimators, self.out_dims))
        self.weights[1:, :] *= self.learning_rate
        # 迭代过程
        for i in self.progressbar(range(self.n_estimators)):
            for c in range(self.out_dims):
                tree = XGBoostRegressionTree(
                        min_samples_split=self.min_samples_split,
                        min_impurity=self.min_impurity,
                        max_depth=self.max_depth,
                        loss=self.loss,
                        gamma=self.gamma,
                        lambd=self.lambd)
                # 计算损失的梯度，并用梯度进行训练
                if not self.is_regression:   
                    Y_hat = softmax(Y_pred)
                    y, y_pred = Y[:, c], Y_hat[:, c]
                else:
                    y, y_pred = Y[:, c], Y_pred[:, c]

                y, y_pred = y.reshape(-1, 1), y_pred.reshape(-1, 1)
                y_and_ypred = np.concatenate((y, y_pred), axis=1)
                tree.fit(X, y_and_ypred)
                # 用新的基学习器进行预测
                h_pred = tree.predict(X)
                # 加法模型中添加基学习器的预测，得到最新迭代下的加法模型预测
                Y_pred[:, c] += np.multiply(self.weights[i, c], h_pred)
                self.trees[i, c] = tree

    def predict(self, X):
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        # 生成预测
        for c in range(self.out_dims):
            y_pred = np.array([])
            for i in range(self.n_estimators):
                update = np.multiply(self.weights[i, c], self.trees[i, c].predict(X))
                y_pred = update if not y_pred.any() else y_pred + update
            Y_pred[:, c] = y_pred
        if not self.is_regression: 
            # 分类问题输出最可能类别
            Y_pred = Y_pred.argmax(axis=1)
        return Y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy
    
    
class XGBRegressor(XGBoost):
    
    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), is_regression=True,
                 gamma=0., lambd=0.):
        super(XGBRegressor, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_impurity,
            max_depth=max_depth,
            is_regression=is_regression,
            gamma=gamma,
            lambd=lambd)


class XGBClassifier(XGBoost):
    
    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), is_regression=False,
                 gamma=0., lambd=0.):
        super(XGBClassifier, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_impurity,
            max_depth=max_depth,
            is_regression=is_regression,
            gamma=gamma,
            lambd=lambd)        
