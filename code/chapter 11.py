import pandas as pd
import numpy as np
import itertools
import time
import re
from scipy.stats import norm
import matplotlib.pyplot as plt


def cal_conf_matrix(labels, preds):
    """
    计算混淆矩阵。
    
    参数说明：
    labels：样本标签 (真实结果)
    preds：预测结果
    """
    n_sample = len(labels)
    result = pd.DataFrame(index=range(0,n_sample),columns=('probability','label'))
    result['label'] = np.array(labels)
    result['probability'] = np.array(preds)
    cm = np.arange(4).reshape(2,2)
    cm[0,0] = len(result[result['label']==1][result['probability']>=0.5]) # TP，注意这里是以 0.5 为阈值
    cm[0,1] = len(result[result['label']==1][result['probability']<0.5])  # FN
    cm[1,0] = len(result[result['label']==0][result['probability']>=0.5]) # FP
    cm[1,1] = len(result[result['label']==0][result['probability']<0.5])  # TN  
    return cm


def cal_PRF1(labels, preds):
    """
    计算查准率P，查全率R，F1值。
    """
    cm = cal_conf_matrix(labels, preds)
    P = cm[0,0]/(cm[0,0]+cm[1,0])
    R = cm[0,0]/(cm[0,0]+cm[0,1])
    F1 = 2*P*R/(P+R)
    return P, R, F1


def cal_PRcurve(labels, preds):
    """
    计算PR曲线上的值。
    """
    n_sample = len(labels)
    result = pd.DataFrame(index=range(0,n_sample),columns=('probability','label'))
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    result['label'] = np.array(labels)
    result['probability'] = np.array(preds)
    result.sort_values('probability',inplace=True,ascending=False)
    PandR = pd.DataFrame(index=range(len(labels)),columns=('P','R'))
    for j in range(len(result)):
        # 以每一个概率为分类的阈值，统计此时正例和反例的数量
        result_j = result.head(n=j+1)
        P = len(result_j[result_j['label']==1])/float(len(result_j))  # 当前实际为正的数量/当前预测为正的数量
        R = len(result_j[result_j['label']==1])/float(len(result[result['label']==1]))  # 当前真正例的数量/实际为正的数量
        PandR.iloc[j] = [P,R]
    return PandR


def cal_ROCcurve(labels, preds):
    """
    计算ROC曲线上的值。
    """
    n_sample = len(labels)
    result = pd.DataFrame(index=range(0,n_sample),columns=('probability','label'))
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    result['label'] = np.array(labels)
    result['probability'] = np.array(preds)
    # 计算 TPR,FPR
    result.sort_values('probability',inplace=True,ascending=False)
    TPRandFPR=pd.DataFrame(index=range(len(result)),columns=('TPR','FPR'))
    for j in range(len(result)):
        # 以每一个概率为分类的阈值，统计此时正例和反例的数量
        result_j=result.head(n=j+1)
        TPR=len(result_j[result_j['label']==1])/float(len(result[result['label']==1]))  # 当前真正例的数量/实际为正的数量
        FPR=len(result_j[result_j['label']==0])/float(len(result[result['label']==0]))  # 当前假正例的数量/实际为负的数量
        TPRandFPR.iloc[j]=[TPR,FPR]
    return TPRandFPR


def timeit(func):
    """
    装饰器，计算函数执行时间
    """
    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        exec_time = time_end - time_start
        print("{function} exec time: {time}s".format(function=func.__name__,time=exec_time))
        return result
    return wrapper

@timeit
def area_auc(labels, preds):
    """
    AUC值的梯度法计算
    """
    TPRandFPR = cal_ROCcurve(labels, preds)
    # 计算AUC，计算小矩形的面积之和
    auc = 0.
    prev_x = 0
    for x, y in zip(TPRandFPR.FPR,TPRandFPR.TPR):
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x
    return auc

@timeit
def naive_auc(labels, preds):
    """
    AUC值的概率法计算
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg  # 总的正负样本对的数目
    labels_preds = zip(labels, preds)
    labels_preds = sorted(labels_preds,key=lambda x:x[1])  # 对预测概率升序排序
    count_neg = 0  # 统计负样本出现的个数
    satisfied_pair = 0   # 统计满足条件的样本对的个数
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += count_neg  # 表明在这个正样本下，有哪些负样本满足条件
        else:
            count_neg += 1
    return satisfied_pair / float(total_pair)


#####----Bayesian Hyperparameter Optimization----####
class KernelBase(ABC):
    
    def __init__(self):
        super().__init__()
        self.params = {}
        self.hyperparams = {}

    @abstractmethod
    def _kernel(self, X, Y):
        raise NotImplementedError

    def __call__(self, X, Y=None):
        return self._kernel(X, Y)

    def __str__(self):
        P, H = self.params, self.hyperparams
        p_str = ", ".join(["{}={}".format(k, v) for k, v in P.items()])
        return "{}({})".format(H["op"], p_str)

    def summary(self):
        return {
            "op": self.hyperparams["op"],
            "params": self.params,
            "hyperparams": self.hyperparams,
        }


class RBFKernel(KernelBase):
    
    def __init__(self, sigma=None):
        """
        RBF 核。
        """
        super().__init__()
        self.hyperparams = {"op": "RBFKernel"}
        self.params = {"sigma": sigma}  # 如果 sigma 未赋值则默认为 np.sqrt(n_features/2)，n_features 为特征数。

    def _kernel(self, X, Y=None):
        """
        对 X 和 Y 的行的每一对计算 RBF 核。如果 Y 为空，则 Y=X。

        参数说明：
        X：输入数组，为 (n_samples, n_features)
        Y：输入数组，为 (m_samples, n_features)
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = X if Y is None else Y
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        assert X.ndim == 2 and Y.ndim == 2, "X and Y must have 2 dimensions"
        sigma = np.sqrt(X.shape[1] / 2) if self.params["sigma"] is None else self.params["sigma"]
        X, Y = X / sigma, Y / sigma
        D = -2 * X @ Y.T + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
        D[D < 0] = 0
        return np.exp(-0.5 * D)
    

class KernelInitializer(object):
    
    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        r = r"([a-zA-Z0-9]*)=([^,)]*)"
        kr_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, self.param)])
        if "rbf" in kr_str:
            kernel = RBFKernel(**kwargs)
        else:
            raise NotImplementedError("{}".format(kr_str))
        return kernel


class GPRegression:
    """
    高斯过程回归
    """
    def __init__(self, kernel="RBFKernel", sigma=1e-10):
        self.kernel = KernelInitializer(kernel)()
        self.params = {"GP_mean": None, "GP_cov": None, "X": None}
        self.hyperparams = {"kernel": str(self.kernel), "sigma": sigma}

    def fit(self, X, y):
        """
        用已有的样本集合得到 GP 先验。

        参数说明：
        X：输入数组，为 (n_samples, n_features)
        y：输入数组 X 的目标值，为 (n_samples)
        """
        mu = np.zeros(X.shape[0])
        Cov = self.kernel(X, X)
        self.params["X"] = X
        self.params["y"] = y
        self.params["GP_cov"] = Cov
        self.params["GP_mean"] = mu

    def predict(self, X_star, conf_interval=0.95):
        """
        对新的样本 X 进行预测。

        参数说明：
        X_star：输入数组，为 (n_samples, n_features)
        conf_interval：置信区间，浮点型 (0, 1)，default=0.95
        """
        X = self.params["X"]
        y = self.params["y"]
        K = self.params["GP_cov"]
        sigma = self.hyperparams["sigma"]
        K_star = self.kernel(X_star, X)
        K_star_star = self.kernel(X_star, X_star)
        sig = np.eye(K.shape[0]) * sigma
        K_y_inv = np.linalg.pinv(K + sig)
        mean = K_star @ K_y_inv @ y
        cov = K_star_star - K_star @ K_y_inv @ K_star.T
        percentile = norm.ppf(conf_interval)
        conf = percentile * np.sqrt(np.diag(cov))
        return mean, conf, cov


class BayesianOptimization:
    
    def __init__(self):
        self.model = GPRegression()
        
    def acquisition_function(self, Xsamples):
        mu, _, cov = self.model.predict(Xsamples)
        mu = mu if mu.ndim==1 else (mu.T)[0]
        ysample = np.random.multivariate_normal(mu, cov) 
        return ysample
    
    def opt_acquisition(self, X, n_samples=20):
        # 样本搜索策略，一般方法有随机搜索、基于网格的搜索，或局部搜索
        # 我们这里就用简单的随机搜索，这里也可以定义样本的范围
        Xsamples = np.random.randint(low=1,high=50,size=n_samples*X.shape[1])
        Xsamples = Xsamples.reshape(n_samples, X.shape[1])
        # 计算采集函数的值并取最大的值
        scores = self.acquisition_function(Xsamples)
        ix = np.argmax(scores)
        return Xsamples[ix, 0]
    
    def fit(self, f, X, y):
        # 拟合 GPR 模型
        self.model.fit(X, y)
        # 优化过程
        for i in range(15):
            x_star = self.opt_acquisition(X)  # 下一个采样点
            y_star = f(x_star)
            mean, conf, cov = self.model.predict(np.array([[x_star]]))
            # 添加当前数据到数据集合
            X = np.vstack((X, [[x_star]]))
            y = np.vstack((y, [[y_star]]))
            # 更新 GPR 模型
            self.model.fit(X, y)
        ix = np.argmax(y)
        print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
        return X[ix], y[ix]    

