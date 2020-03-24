import numpy as np
import cvxopt
import math


########-----NaiveBayes------#########
class NaiveBayes():
    
    def __init__(self):
        self.parameters = [] # 保存每个特征针对每个类的均值和方差
        self.y = None
        self.classes = None

    def fit(self, X, y):
        self.y = y
        self.classes = np.unique(y) # 类别 
        # 计算每个特征针对每个类的均值和方差
        for i, c in enumerate(self.classes):
            # 选择类别为c的X
            X_where_c = X[np.where(self.y == c)]
            self.parameters.append([])
            # 添加均值与方差
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)
    
    def _calculate_prior(self, c):
        """
        先验函数。
        """
        frequency = np.mean(self.y == c)
        return frequency

    def _calculate_likelihood(self, mean, var, X):
        """
        似然函数。
        """
        # 高斯概率
        eps = 1e-4 # 防止除数为0
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(X - mean, 2) / (2 * var + eps)))
        return coeff * exponent
    
    def _calculate_probabilities(self, X):
        posteriors = []
        for i, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(X, self.parameters[i]):
                # 独立性假设
                # P(x1,x2|Y) = P(x1|Y)*P(x2|Y)
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        # 返回具有最大后验概率的类别
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        y_pred = [self._calculate_probabilities(sample) for sample in X]
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy


########-----LogisticRegression------#########
def Sigmoid(x):
    return 1/(1 + np.exp(-x))

class LogisticRegression():

    def __init__(self, learning_rate=.1):
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # 初始化参数theta， [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # 参数theta的迭代更新
        for i in range(n_iterations):
            # 求预测
            y_pred = self.sigmoid(X.dot(self.param))
            # 最小化损失函数，参数更新公式
            self.param -= self.learning_rate * -(y - y_pred).dot(X)

    def predict(self, X):
        y_pred = self.sigmoid(X.dot(self.param))
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy
    

########-----SupportVectorMachine------#########
# 隐藏cvxopt输出
cvxopt.solvers.options['show_progress'] = False

def linear_kernel(**kwargs):
    """
    线性核
    """
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power, coef, **kwargs):
    """
    多项式核
    """
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

def rbf_kernel(gamma, **kwargs):
    """
    高斯核
    """
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f

class SupportVectorMachine():

    def __init__(self, kernel=linear_kernel, power=4, gamma=None, coef=4):
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        # gamma默认设置为1 / n_features
        if not self.gamma:
            self.gamma = 1 / n_features
        
        # 定义核函数
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        # 计算Gram矩阵
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])
        
        # 构造二次规划问题
        # 形式为 min (1/2)x.T*P*x+q.T*x, s.t. G*x<=h, A*x=b
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        G = cvxopt.matrix(np.identity(n_samples) * -1)
        h = cvxopt.matrix(np.zeros(n_samples))

        # 用cvxopt求解二次规划问题
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        lagr_mult = np.ravel(minimization['x'])
        # 非0的alpha值
        idx = lagr_mult > 1e-7
        # alpha值
        self.lagr_multipliers = lagr_mult[idx]
        # 支持向量
        self.support_vectors = X[idx]
        # 支持向量的标签
        self.support_vector_labels = y[idx]

        # 通过第一个支持向量计算b
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        for sample in X:
            # 对于输入的x, 计算f(x)
            prediction = 0
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

    
########-----KNN------#########
class KNN():
    
    def __init__(self, k=10):
        self._k = k

    def fit(self, X, y):
        self._unique_labels = np.unique(y)
        self._class_num = len(self._unique_labels)
        self._datas = X
        self._labels = y.astype(np.int32)

    def predict(self, X):
        # 欧式距离计算
        dist = np.sum(np.square(X), axis=1, keepdims=True) - 2 * np.dot(X, self._datas.T)
        dist = dist + np.sum(np.square(self._datas), axis=1, keepdims=True).T
        dist = np.argsort(dist)[:,:self._k]
        return np.array([np.argmax(np.bincount(self._labels[dist][i])) for i in range(len(X))])
        idx = lagr_mult > 1e-7
        # alpha值
        self.lagr_multipliers = lagr_mult[idx]
        # 支持向量
        self.support_vectors = X[idx]
        # 支持向量的标签
        self.support_vector_labels = y[idx]

        # 通过第一个支持向量计算b
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        for sample in X:
            # 对于输入的x, 计算f(x)
            prediction = 0
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

    
########-----DecisionTree------#########
class DecisionNode():

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # 当前结点测试的特征的索引
        self.threshold = threshold          # 当前结点测试的特征的阈值
        self.value = value                  # 结点值（如果结点为叶子结点）
        self.true_branch = true_branch      # 左子树（满足阈值， 将特征值大于等于切分点值的数据划分为左子树）
        self.false_branch = false_branch    # 右子树（未满足阈值， 将特征值小于切分点值的数据划分为右子树）

        
def divide_on_feature(X, feature_i, threshold):
    """
    依据切分变量和切分点，将数据集分为两个子区域
    """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


class DecisionTree(object):

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # 根结点
        self.min_samples_split = min_samples_split  # 满足切分的最少样本数
        self.min_impurity = min_impurity  # 满足切分的最小纯度
        self.max_depth = max_depth  # 树的最大深度
        self._impurity_calculation = None  # 计算纯度的函数，如对于分类树采用信息增益
        self._leaf_value_calculation = None  # 计算y在叶子结点值的函数
        self.one_dim = None  # y是否为one-hot编码

    def fit(self, X, y):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        """
        递归方法建立决策树
        """
        largest_impurity = 0
        best_criteria = None    # 当前最优分类的特征索引和阈值
        best_sets = None        # 数据子集

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # 对每个特征计算纯度
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # 遍历特征i所有的可能值找到最优纯度
                for threshold in unique_values:
                    # 基于X在特征i处是否满足阈值来划分X和y， Xy1为满足阈值的子集
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # 取出Xy中y的集合
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # 计算纯度
                        impurity = self._impurity_calculation(y, y1, y2)

                        # 如果纯度更高，则更新
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X的左子树
                                "lefty": Xy1[:, n_features:],   # y的左子树
                                "rightX": Xy2[:, :n_features],  # X的右子树
                                "righty": Xy2[:, n_features:]   # y的右子树
                                }

        if largest_impurity > self.min_impurity:
            # 建立左子树和右子树
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # 如果是叶结点则计算值
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)


    def predict_value(self, x, tree=None):
        """
        预测样本，沿着树递归搜索
        """
        # 根结点
        if tree is None:
            tree = self.root

        # 递归出口
        if tree.value is not None:
            return tree.value

        # 选择当前结点的特征
        feature_value = x[tree.feature_i]

        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy
    
    def print_tree(self, tree=None, indent=" "):
        """
        输出树
        """
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("feature|threshold -> %s | %s" % (tree.feature_i, tree.threshold))
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def calculate_gini(y):
    unique_labels = np.unique(y)
    var = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        var += p ** 2
    return 1 - var


class ClassificationTree(DecisionTree):
    """
    分类树，在决策书节点选择计算信息增益/基尼指数，在叶子节点选择多数表决。
    """
    def _calculate_gini_index(self, y, y1, y2):
        """
        计算基尼指数
        """
        p = len(y1) / len(y)
        gini = calculate_gini(y)
        gini_index = gini - p * \
            calculate_gini(y1) - (1 - p) * \
            calculate_gini(y2)
        return gini_index
    
    
    def _calculate_information_gain(self, y, y1, y2):
        """
        计算信息增益
        """
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
            calculate_entropy(y1) - (1 - p) * \
            calculate_entropy(y2)
        return info_gain

    def _majority_vote(self, y):
        """
        多数表决
        """
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_gini_index
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


def calculate_mse(y):
    return np.mean((y - np.mean(y)) ** 2)


def calculate_variance(y):
    n_samples = np.shape(y)[0]
    variance = (1 / n_samples) * np.diag((y - np.mean(y)).T.dot(y - np.mean(y)))
    return variance


class RegressionTree(DecisionTree):
    """
    回归树，在决策书节点选择计算MSE/方差降低，在叶子节点选择均值。
    """
    def _calculate_mse(self, y, y1, y2):
        """
        计算MSE降低
        """
        mse_tot = calculate_mse(y)
        mse_1 = calculate_mse(y1)
        mse_2 = calculate_mse(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        mse_reduction = mse_tot - (frac_1 * mse_1 + frac_2 * mse_2)
        return mse_reduction
    
    def _calculate_variance_reduction(self, y, y1, y2):
        """
        计算方差降低
        """
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        """
        计算均值
        """
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_mse
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)


########-----PCA------#########
class PCA():
    
    def __init__(self):
        pass
    
    def fit(self, X, n_components):
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))

        # 对协方差矩阵进行特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 对特征值（特征向量）从大到小排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # 得到低维表示
        X_transformed = X.dot(eigenvectors)


########-----KMeans------#########
def distEclud(x,y):
    """
    计算欧氏距离
    """
    return np.sqrt(np.sum((x-y)**2))  

def randomCent(dataSet,k):
    """
    为数据集构建一个包含 K 个随机质心的集合
    """
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m))
        centroids[i,:] = dataSet[index,:]
    return centroids

class KMeans():
    
    def __init__(self):
        self.dataSet = None
        self.k = None
        
    def fit(self, dataSet, k):
        self.dataSet = dataSet
        self.k = k
        m = np.shape(dataSet)[0]
        # 第一列存样本属于哪一簇
        # 第二列存样本的到簇的中心点的误差
        clusterAssment = np.mat(np.zeros((m,2)))
        clusterChange = True
        centroids = randomCent(self.dataSet,k)
        while clusterChange:
            clusterChange = False
            for i in range(m):
                minDist = 1e6
                minIndex = -1
                # 遍历所有的质心, 找出最近的质心
                for j in range(k):
                    distance = distEclud(centroids[j,:], self.dataSet[i,:])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                # 更新每一行样本所属的簇
                if clusterAssment[i,0] != minIndex:
                    clusterChange = True
                    clusterAssment[i,:] = minIndex, minDist**2
            # 更新质心
            for j in range(k):
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
                centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值

        return centroids,clusterAssment

        return X_transformed
