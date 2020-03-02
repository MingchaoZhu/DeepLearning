from abc import ABC, abstractmethod
import numpy as np
from chapter6 import LayerBase, CrossEntropy, FullyConnected, minibatch, softmax
from collections import OrderedDict


########## Padding ################
def calc_pad_dims_sameconv_2D(X_shape, out_dim, kernel_shape, stride, dilation=1):
    """
    当填充方式为相同卷积时，计算 padding 的数目，保证输入输出的大小相同。这里在卷积过程中考虑填充(Padding)，
    卷积步幅(Stride)，扩张率(Dilation rate)。根据扩张卷积的输出公式可以得到 padding 的数目。
    
    参数说明：
    X_shape：输入数组，为 (n_samples, in_rows, in_cols, in_ch)
    out_dim：输出数组维数，为 (out_rows, out_cols)
    kernel_shape：卷积核形状，为 (fr, fc)
    stride：卷积步幅，int 型
    dilation：扩张率，int 型，default=1
    """
    d = dilation
    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, in_ch = X_shape

    # 考虑扩张率
    _fr, _fc = fr + (fr-1) * (d-1), fc + (fc-1) * (d-1)

    # 计算 padding 维数
    pr = int((stride * (out_rows-1) + _fr - in_rows) / 2)
    pc = int((stride * (out_cols-1) + _fc - in_cols) / 2)

    # 校验，如不等 (right/bottom处) 添加不对称0填充
    out_rows1 = int(1 + (in_rows + 2 * pr - _fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - _fc) / stride)
    
    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError
        
    # 返回对 X 的 Padding 维数 (left, right, up, down)
    return (pr1, pr2, pc1, pc2)


def pad2D(X, pad, kernel_shape=None, stride=None, dilation=1):
    """
    二维填充
    
    参数说明：
    X：输入数组，为 (n_samples, in_rows, in_cols, in_ch)，
        其中 padding 操作是应用到 in_rows 和 in_cols
    pad：padding 数目，4-tuple, int, 或 'same'，'valid'
        在图片的左、右、上、下 (left, right, up, down) 0填充
        若为int，表示在左、右、上、下均填充数目为 pad 的 0，
        若为same，表示填充后为相同 (same) 卷积，
        若为valid，表示填充后为有效 (valid) 卷积
    kernel_shape：卷积核形状，为 (fr, fc)
    stride：卷积步幅，int 型
    dilation：扩张率，int 型，default=1
    """
    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # 'same'卷积，首先计算 padding 维数
    if p == "same" and kernel_shape and stride is not None:
        p = calc_pad_dims_sameconv_2D(
            X.shape, X.shape[1:3], kernel_shape, stride, dilation=dilation
        )
        X_pad, p = pad2D(X, p)
        
    if p == "valid":
        p = (0, 0, 0, 0)
        X_pad, p = pad2D(X, p)
        
    return X_pad, p


####### conv2D ##################
def conv2D(X, W, stride, pad, dilation=1):
    """
    二维卷积实现过程。

    参数说明：
    X：输入数组，为 (n_samples, in_rows, in_cols, in_ch)
    W：卷积层的卷积核参数，为 (kernel_rows, kernel_cols, in_ch, out_ch)
    stride：卷积核的卷积步幅，int型
    pad：padding 数目，4-tuple, int, 或 'same'，'valid'型
        在图片的左、右、上、下 (left, right, up, down) 0填充
        若为int，表示在左、右、上、下均填充数目为 pad 的 0，
        若为same，表示填充后为相同 (same) 卷积，
        若为valid，表示填充后为有效 (valid) 卷积
    dilation：扩张率，int 型，default=1

    输出说明：
    Z：卷积结果，为 (n_samples, out_rows, out_cols, out_ch)
    """
    s, d = stride, dilation
    X_pad, p = pad2D(X, pad, W.shape[:2], stride=s, dilation=d)

    pr1, pr2, pc1, pc2 = p
    fr, fc, in_ch, out_ch = W.shape
    n_samp, in_rows, in_cols, in_ch = X.shape

    # 考虑扩张率
    _fr, _fc = fr + (fr-1) * (d-1), fc + (fc-1) * (d-1)

    out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

    Z = np.zeros((n_samp, out_rows, out_cols, out_ch))
    for m in range(n_samp):
        for c in range(out_ch):
            for i in range(out_rows):
                for j in range(out_cols):
                    i0, i1 = i * s, (i * s) + fr + (fr-1) * (d-1)
                    j0, j1 = j * s, (j * s) + fc + (fc-1) * (d-1)

                    window = X_pad[m, i0 : i1 : d, j0 : j1 : d, :]
                    Z[m, i, j, c] = np.sum(window * W[:, :, :, c])
    return Z


####### conv2D GEMM ############
"""
conv2D 的 GEMM 实现过程，将 X 和 W 转化为 2D 矩阵，
这里我们将 X 转化为 (kernel_rows*kernel_cols*in_ch, n_samples*out_rows*out_cols)
W 转化为 (out_ch, kernel_rows*kernel_cols*in_ch)
"""
def _im2col_indices(X_shape, fr, fc, p, s, d=1):
    """
    生成输入矩阵的 (c,h_in,w_in) 三个维度的索引
    
    输出说明：
    i：输入矩阵的i值，(kernel_rows*kernel_cols*in_ch, out_rows*out_cols)，图示中第二维坐标
    j：输入矩阵的j值，(kernel_rows*kernel_cols*in_ch, out_rows*out_cols)，图示中第三维坐标
    k：输入矩阵的c值，(kernel_rows*kernel_cols*in_ch, 1)，图示中第一维坐标
    """
    pr1, pr2, pc1, pc2 = p
    n_ex, n_in, in_rows, in_cols = X_shape

    # 考虑扩张率
    _fr, _fc = fr + (fr-1) * (d-1), fc + (fc-1) * (d-1)

    out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

    # i0/i1/j0/j1：用于得到i，j，k。i0/j0过程见图示，i1/j1由滑动过程得出
    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, n_in) * d
    i1 = s * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * n_in) * d
    j1 = s * np.tile(np.arange(out_cols), out_rows)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
    return k, i, j



def im2col(X, W_shape, pad, stride, dilation=1):
    """
    im2col 实现

    参数说明：
    X：输入数组，为 (n_samples, in_rows, in_cols, in_ch)，此时还未 0 填充(padding)
    W_shape：卷积层的卷积核的形状，为 (kernel_rows, kernel_cols, in_ch, out_ch)
    pad：padding 数目，4-tuple, int, 或 'same'，'valid'型
        在图片的左、右、上、下 (left, right, up, down) 0填充
        若为int，表示在左、右、上、下均填充数目为 pad 的 0，
        若为same，表示填充后为相同 (same) 卷积，
        若为valid，表示填充后为有效 (valid) 卷积
    stride：卷积核的卷积步幅，int型
    dilation：扩张率，int 型，default=1

    输出说明：
    X_col：输出结果，形状为 (kernel_rows*kernel_cols*n_in, n_samples*out_rows*out_cols)
    p：填充数，4-tuple
    """
    fr, fc, n_in, n_out = W_shape
    s, p, d = stride, pad, dilation
    n_samp, in_rows, in_cols, n_in = X.shape

    X_pad, p = pad2D(X, p, W_shape[:2], stride=s, dilation=d)
    pr1, pr2, pc1, pc2 = p

    # 将输入的通道维数移至第二位
    X_pad = X_pad.transpose(0, 3, 1, 2)

    k, i, j = _im2col_indices((n_samp, n_in, in_rows, in_cols), fr, fc, p, s, d)

    # X_col.shape = (n_samples, kernel_rows*kernel_cols*n_in, out_rows*out_cols)
    X_col = X_pad[:, k, i, j]
    X_col = X_col.transpose(1, 2, 0).reshape(fr * fc * n_in, -1)
    return X_col, p


def conv2D_gemm(X, W, stride=0, pad='same', dilation=1):
    """
    二维卷积实现过程，依靠“im2col”函数将卷积作为单个矩阵乘法执行。

    参数说明：
    X：输入数组，为 (n_samples, in_rows, in_cols, in_ch)
    W：卷积层的卷积核参数，为 (kernel_rows, kernel_cols, in_ch, out_ch)
    stride：卷积核的卷积步幅，int型
    pad：padding 数目，4-tuple, int, 或 'same'，'valid'型
        在图片的左、右、上、下 (left, right, up, down) 0填充
        若为int，表示在左、右、上、下均填充数目为 pad 的 0，
        若为same，表示填充后为相同 (same) 卷积，
        若为valid，表示填充后为有效 (valid) 卷积
    dilation：扩张率，int 型，default=1

    输出说明：
    Z：卷积结果，为 (n_samples, out_rows, out_cols, out_ch)
    """
    s, d = stride, dilation
    _, p = pad2D(X, pad, W.shape[:2], s, dilation=dilation)

    pr1, pr2, pc1, pc2 = p
    fr, fc, in_ch, out_ch = W.shape
    n_samp, in_rows, in_cols, in_ch = X.shape
    
    # 考虑扩张率
    _fr, _fc = fr + (fr-1) * (d-1), fc + (fc-1) * (d-1)

    # 输出维数，根据上面公式可得
    out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

    # 将 X 和 W 转化为 2D 矩阵并乘积
    X_col, _ = im2col(X, W.shape, p, s, d)
    W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)

    Z = (W_col @ X_col).reshape(out_ch, out_rows, out_cols, n_samp).transpose(3, 1, 2, 0)

    return Z


########### Conv2D ##################
class Conv2D(LayerBase):
    
    def __init__(
        self,
        out_ch,
        kernel_shape,
        pad=0,
        stride=1,
        dilation=1,
        acti_fn=None,
        optimizer=None,
        init_w="glorot_uniform",
    ):
        """
        二维卷积

        参数说明：
        out_ch：卷积核组的数目，int 型
        kernel_shape：单个卷积核形状，2-tuple
        acti_fn：激活函数，str 型
        pad：padding 数目，4-tuple, int, 或 'same'，'valid'型
            在图片的左、右、上、下 (left, right, up, down) 0填充
            若为int，表示在左、右、上、下均填充数目为 pad 的 0，
            若为same，表示填充后为相同 (same) 卷积，
            若为valid，表示填充后为有效 (valid) 卷积
        stride：卷积核的卷积步幅，int型
        dilation：扩张率，int 型，default=1
        init_w：权重初始化方法，str型
        optimizer：优化方法，str型
        """
        super().__init__(optimizer)

        self.pad = pad
        self.in_ch = None
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.kernel_shape = kernel_shape
        self.init_w = init_w
        self.init_weights = WeightInitializer(mode=init_w)
        self.acti_fn = ActivationInitializer(acti_fn)()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        fr, fc = self.kernel_shape
        W = self.init_weights((fr, fc, self.in_ch, self.out_ch))
        b = np.zeros((1, 1, 1, self.out_ch))

        self.params = {"W": W, "b": b}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.derived_variables = {"Y": []}
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        """
        卷积层的前向传播，原理见上文。

        参数说明：
        X：输入数组，形状为 (n_samples, in_rows, in_cols, in_ch)
        retain_derived：是否保留中间变量，以便反向传播时再次使用，bool型

        输出说明：
        a：卷积层输出，形状为 (n_samples, out_rows, out_cols, out_ch)
        """
        if not self.is_initialized:
            self.in_ch = X.shape[3]
            self._init_params()

        W = self.params["W"]
        b = self.params["b"]

        n_samp, in_rows, in_cols, in_ch = X.shape
        s, p, d = self.stride, self.pad, self.dilation

        # 卷积操作
        Y = conv2D(X, W, s, p, d) + b
        a = self.acti_fn(Y)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["Y"].append(Y)

        return a

    def backward(self, dLda, retain_grads=True):
        """
        卷积层的反向传播，原理见上文。

        参数说明：
        dLda：关于损失的梯度，为 (n_samples, out_rows, out_cols, out_ch) 
        retain_grads：是否计算中间变量的参数梯度，bool型

        输出说明：
        dXs：即dX，当前卷积层对输入关于损失的梯度，为 (n_samples, in_rows, in_cols, in_ch)
        """
        if not isinstance(dLda, list):
            dLda = [dLda]

        W = self.params["W"]
        b = self.params["b"]
        Ys = self.derived_variables["Y"]
        Xs, d = self.X, self.dilation
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad
        dXs = []
        
        for X, Y, da in zip(Xs, Ys, dLda):
            n_samp, out_rows, out_cols, out_ch = da.shape
            X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s, d)

            dY = da * self.acti_fn.grad(Y)

            dX = np.zeros_like(X_pad)
            dW, db = np.zeros_like(W), np.zeros_like(b)
            for m in range(n_samp):
                for i in range(out_rows):
                    for j in range(out_cols):
                        for c in range(out_ch):
                            i0, i1 = i * s, (i * s) + fr + (fr-1) * (d-1)
                            j0, j1 = j * s, (j * s) + fc + (fc-1) * (d-1)

                            wc = W[:, :, :, c]
                            kernel = dY[m, i, j, c]
                            window = X_pad[m, i0:i1:d, j0:j1:d, :]

                            db[:, :, :, c] += kernel
                            dW[:, :, :, c] += window * kernel
                            dX[m, i0:i1:d, j0:j1:d, :] += (
                                wc * kernel
                            )

            if retain_grads:
                self.gradients["W"] += dW
                self.gradients["b"] += db

            pr2 = None if pr2 == 0 else -pr2
            pc2 = None if pc2 == 0 else -pc2
            dXs.append(dX[:, pr1:pr2, pc1:pc2, :])
            
        return dXs[0] if len(Xs) == 1 else dXs
    
    @property
    def hyperparams(self):
        return {
            "layer": "Conv2D",
            "pad": self.pad,
            "init_w": self.init_w,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "dilation": self.dilation,
            "acti_fn": str(self.acti_fn),
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparams": self.optimizer.hyperparams,
            },
        }


######### Conv2D GEMM #############
def col2im(X_col, X_shape, W_shape, pad, stride, dilation=0):
    """
    col2im 实现，“col2im”函数将 2D 矩阵变为 4D 图像

    参数说明：
    X_col：X 经过 im2col 后 (列) 的矩阵，形状为 (Q, Z)，具体形状见上文
    X_shape：原始的输入数组形状，为 (n_samples, in_rows, in_cols, in_ch)，
             此时还未 0 填充(padding)
    W_shape：卷积核组形状，4-tuple 为 (kernel_rows, kernel_cols, in_ch, out_ch)
    pad：padding 数目，4-tuple
            在图片的左、右、上、下 (left, right, up, down) 0填充
    stride：卷积核的卷积步幅，int型
    dilation：扩张率，int 型，default=1

    输出说明：
    img：输出结果，形状为 (n_samples, in_rows, in_cols, in_ch)
    """
    s, d = stride, dilation
    pr1, pr2, pc1, pc2 = pad
    fr, fc, n_in, n_out = W_shape
    n_samp, in_rows, in_cols, n_in = X_shape

    X_pad = np.zeros((n_samp, n_in, in_rows + pr1 + pr2, in_cols + pc1 + pc2))
    k, i, j = _im2col_indices((n_samp, n_in, in_rows, in_cols), fr, fc, pad, s, d)

    X_col_reshaped = X_col.reshape(n_in * fr * fc, -1, n_samp)
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)

    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    return X_pad[:, :, pr1:pr2, pc1:pc2]


class Conv2D_gemm(LayerBase):
    
    def __init__(
        self,
        out_ch,
        kernel_shape,
        pad=0,
        stride=1,
        dilation=1,
        acti_fn=None,
        optimizer=None,
        init_w="glorot_uniform",
    ):
        """
        二维卷积

        参数说明：
        out_ch：卷积核组的数目，int 型
        kernel_shape：单个卷积核形状，2-tuple
        acti_fn：激活函数，str 型
        pad：padding 数目，4-tuple, int, 或 'same'，'valid'型
            在图片的左、右、上、下 (left, right, up, down) 0填充
            若为int，表示在左、右、上、下均填充数目为 pad 的 0，
            若为same，表示填充后为相同 (same) 卷积，
            若为valid，表示填充后为有效 (valid) 卷积
        stride：卷积核的卷积步幅，int型
        dilation：扩张率，int 型，default=1
        init_w：权重初始化方法，str型
        optimizer：优化方法，str型
        """
        super().__init__(optimizer)

        self.pad = pad
        self.in_ch = None
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.kernel_shape = kernel_shape
        self.init_w = init_w
        self.init_weights = WeightInitializer(mode=init_w)
        self.acti_fn = ActivationInitializer(acti_fn)()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        fr, fc = self.kernel_shape
        W = self.init_weights((fr, fc, self.in_ch, self.out_ch))
        b = np.zeros((1, 1, 1, self.out_ch))

        self.params = {"W": W, "b": b}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.derived_variables = {"Y": []}
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        """
        卷积层的前向传播，原理见上文。

        参数说明：
        X：输入数组，形状为 (n_samples, in_rows, in_cols, in_ch)
        retain_derived：是否保留中间变量，以便反向传播时再次使用，bool型

        输出说明：
        a：卷积层输出，形状为 (n_samples, out_rows, out_cols, out_ch)
        """
        if not self.is_initialized:
            self.in_ch = X.shape[3]
            self._init_params()

        W = self.params["W"]
        b = self.params["b"]

        n_samp, in_rows, in_cols, in_ch = X.shape
        s, p, d = self.stride, self.pad, self.dilation

        # 卷积操作
        Y = conv2D_gemm(X, W, s, p, d) + b
        a = self.acti_fn(Y)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["Y"].append(Y)

        return a

    def backward(self, dLda, retain_grads=True):
        """
        卷积层的反向传播，原理见上文。

        参数说明：
        dLda：关于损失的梯度，为 (n_samples, out_rows, out_cols, out_ch) 
        retain_grads：是否计算中间变量的参数梯度，bool型

        输出说明：
        dX：当前卷积层对输入关于损失的梯度，为 (n_samples, in_rows, in_cols, in_ch)
        """
        if not isinstance(dLda, list):
            dLda = [dLda]

        dX = []
        X = self.X
        Y = self.derived_variables["Y"]

        for da, x, y in zip(dLda, X, Y):
            dx, dw, db = self._bwd(da, x, y)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLda, X, Y):
        W = self.params["W"]
        d = self.dilation
        fr, fc, in_ch, out_ch = W.shape
        n_samp, out_rows, out_cols, out_ch = dLda.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad
        
        dLdy = dLda * self.acti_fn.grad(Y)
        dLdy_col = dLdy.transpose(3, 1, 2, 0).reshape(out_ch, -1)
        W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1).T
        X_col, p = im2col(X, W.shape, p, s, d)

        dW = (dLdy_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)
        db = dLdy_col.sum(axis=1).reshape(1, 1, 1, -1)

        dX_col = W_col @ dLdy_col
        dX = col2im(dX_col, X.shape, W.shape, p, s, d).transpose(0, 2, 3, 1)

        return dX, dW, db
    
    @property
    def hyperparams(self):
        return {
            "layer": "Conv2D",
            "pad": self.pad,
            "init_w": self.init_w,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "dilation": self.dilation,
            "acti_fn": str(self.acti_fn),
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparams": self.optimizer.hyperparams,
            },
        }


######## Pool2D ################
class Pool2D(LayerBase):
    
    def __init__(self, kernel_shape, stride=1, pad=0, mode="max", optimizer=None):
        """
        二维池化

        参数说明：
        kernel_shape：池化窗口的大小，2-tuple
        stride：和卷积类似，窗口在每一个维度上滑动的步长，int型
        pad：padding 数目，4-tuple, int, 或 str('same','valid')型 (default: 0)
            和卷积类似
        mode：池化函数，str型 (default: 'max')，可选{"max","average"}
        optimizer：优化方法，str型
        """
        super().__init__(optimizer)

        self.pad = pad
        self.mode = mode
        self.in_ch = None
        self.out_ch = None
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.is_initialized = False

    def _init_params(self):
        self.derived_variables = {"out_rows": [], "out_cols": []}
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        """
        池化层前向传播

        参数说明：
        X：输入数组，形状为 (n_samp, in_rows, in_cols, in_ch)
        retain_derived：是否保留中间变量，以便反向传播时再次使用，bool型
        
        输出说明：
        Y：输出结果，形状为 (n_samp, out_rows, out_cols, out_ch)
        """
        if not self.is_initialized:
            self.in_ch = self.out_ch = X.shape[3]
            self._init_params()

        n_samp, in_rows, in_cols, nc_in = X.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad
        X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        if self.mode == "max":
            pool_fn = np.max
        elif self.mode == "average":
            pool_fn = np.mean

        Y = np.zeros((n_samp, out_rows, out_cols, self.out_ch))
        for m in range(n_samp):
            for i in range(out_rows):
                for j in range(out_cols):
                    for c in range(self.out_ch):
                        i0, i1 = i * s, (i * s) + fr
                        j0, j1 = j * s, (j * s) + fc

                        xi = X_pad[m, i0:i1, j0:j1, c]
                        Y[m, i, j, c] = pool_fn(xi)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["out_rows"].append(out_rows)
            self.derived_variables["out_cols"].append(out_cols)

        return Y

    def backward(self, dLdy, retain_grads=True):
        """
        池化层的反向传播，原理见上文。

        参数说明：
        dLdy：关于损失的梯度，为 (n_samples, out_rows, out_cols, out_ch) 
        retain_grads：是否计算中间变量的参数梯度，bool型

        输出说明：
        dXs：即dX，当前卷积层对输入关于损失的梯度，为 (n_samples, in_rows, in_cols, in_ch)
        """
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        Xs = self.X
        out_rows = self.derived_variables["out_rows"]
        out_cols = self.derived_variables["out_cols"]

        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad

        dXs = []
        for X, dy, out_row, out_col in zip(Xs, dLdy, out_rows, out_cols):
            n_samp, in_rows, in_cols, nc_in = X.shape
            X_pad, (pr1, pr2, pc1, pc2) = pad2D(X, p, self.kernel_shape, s)

            dX = np.zeros_like(X_pad)
            for m in range(n_samp):
                for i in range(out_row):
                    for j in range(out_col):
                        for c in range(self.out_ch):
                            i0, i1 = i * s, (i * s) + fr
                            j0, j1 = j * s, (j * s) + fc

                            if self.mode == "max":
                                xi = X[m, i0:i1, j0:j1, c]
                                mask = np.zeros_like(xi).astype(bool)
                                x, y = np.argwhere(xi == np.max(xi))[0]
                                mask[x, y] = True
                                dX[m, i0:i1, j0:j1, c] += mask * dy[m, i, j, c]
                                
                            elif self.mode == "average":
                                frame = np.ones((fr, fc)) * dy[m, i, j, c]
                                dX[m, i0:i1, j0:j1, c] += frame / np.prod((fr, fc))

            pr2 = None if pr2 == 0 else -pr2
            pc2 = None if pc2 == 0 else -pc2
            dXs.append(dX[:, pr1:pr2, pc1:pc2, :])
            
        return dXs[0] if len(Xs) == 1 else dXs

    @property
    def hyperparams(self):
        return {
            "layer": "Pool2D",
            "acti_fn": None,
            "pad": self.pad,
            "mode": self.mode,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparams": self.optimizer.hyperparams,
            },
        }


############### Flatten ##################
class Flatten(LayerBase):
    
    def __init__(self, keep_dim="first", optimizer=None):
        """
        将多维输入展开

        参数说明：
        keep_dim：展开形状，str (default : 'first')
                对于输入 X，keep_dim可选 'first'->将 X 重构为(X.shape[0], -1)，
                'last'->将 X 重构为(-1, X.shape[0])，'none'->将 X 重构为(1,-1)
        optimizer：优化方法
        """
        super().__init__(optimizer)

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.params = {}
        self.derived_variables = {"in_dims": []}

    def forward(self, X, retain_derived=True):
        """
        前向传播

        参数说明：
        X：输入数组
        retain_derived：是否保留中间变量，以便反向传播时再次使用，bool型
        """
        if retain_derived:
            self.derived_variables["in_dims"].append(X.shape)
        if self.keep_dim == "none":
            return X.flatten().reshape(1, -1)
        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdy, retain_grads=True):
        """
        反向传播

        参数说明：
        dLdy：关于损失的梯度
        retain_grads：是否计算中间变量的参数梯度，bool型

        输出说明：
        dX：将对输入的梯度进行重构为原始输入的形状
        """
        if not isinstance(dLdy, list):
            dLdy = [dLdy]
        in_dims = self.derived_variables["in_dims"]
        dX = [dy.reshape(*dims) for dy, dims in zip(dLdy, in_dims)]
        return dX[0] if len(dLdy) == 1 else dX

    @property
    def hyperparams(self):
        return {
            "layer": "Flatten",
            "keep_dim": self.keep_dim,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparams": self.optimizer.hyperparams,
            },
        }


########### LeNet ################
class LeNet(object):
    
    def __init__(
        self,
        fc3_out=128,
        fc4_out=84,
        fc5_out=10,
        conv1_pad=0,
        conv2_pad=0,
        conv1_out_ch=6,
        conv2_out_ch=16,
        conv1_stride=1,
        pool1_stride=2,
        conv2_stride=1,
        pool2_stride=2,
        conv1_kernel_shape=(5, 5),
        pool1_kernel_shape=(2, 2),
        conv2_kernel_shape=(5, 5),
        pool2_kernel_shape=(2, 2),
        optimizer="adam",
        init_w="glorot_normal",
        loss=CrossEntropy()
    ):
        self.optimizer = optimizer
        self.init_w = init_w
        self.loss = loss
        self.fc3_out = fc3_out
        self.fc4_out = fc4_out
        self.fc5_out = fc5_out
        self.conv1_pad = conv1_pad
        self.conv2_pad = conv2_pad
        self.conv1_stride = conv1_stride
        self.conv1_out_ch = conv1_out_ch
        self.pool1_stride = pool1_stride
        self.conv2_out_ch = conv2_out_ch
        self.conv2_stride = conv2_stride
        self.pool2_stride = pool2_stride
        self.conv2_kernel_shape = conv2_kernel_shape
        self.pool2_kernel_shape = pool2_kernel_shape
        self.conv1_kernel_shape = conv1_kernel_shape
        self.pool1_kernel_shape = pool1_kernel_shape
        
        self.is_initialized = False
    
    def _set_params(self):
        """
        函数作用：模型初始化
        Conv1 -> Pool1 -> Conv2 -> Pool2 -> Flatten -> FC3 -> FC4 -> FC5 -> Softmax
        """
        self.layers = OrderedDict()
        self.layers["Conv1"] = Conv2D(
            out_ch=self.conv1_out_ch,
            kernel_shape=self.conv1_kernel_shape,
            pad=self.conv1_pad,
            stride=self.conv1_stride,
            acti_fn="sigmoid",
            optimizer=self.optimizer,
            init_w=self.init_w,
        )
        self.layers["Pool1"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.pool1_stride,
            kernel_shape=self.pool1_kernel_shape,
        )
        self.layers["Conv2"] = Conv2D(
            out_ch=self.conv1_out_ch,
            kernel_shape=self.conv1_kernel_shape,
            pad=self.conv1_pad,
            stride=self.conv1_stride,
            acti_fn="sigmoid",
            optimizer=self.optimizer,
            init_w=self.init_w,
        )
        self.layers["Pool2"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.pool2_stride,
            kernel_shape=self.pool2_kernel_shape,
        )
        self.layers["Flatten"] = Flatten(optimizer=self.optimizer)
        self.layers["FC3"] = FullyConnected(
            n_out=self.fc3_out,
            acti_fn="sigmoid",
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.layers["FC4"] = FullyConnected(
            n_out=self.fc4_out,
            acti_fn="sigmoid",
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.layers["FC5"] = FullyConnected(
            n_out=self.fc5_out,
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
            "fc3_out": self.fc3_out, 
            "fc4_out": self.fc4_out,
            "fc5_out": self.fc5_out,
            "conv1_pad": self.conv1_pad, 
            "conv2_pad": self.conv2_pad, 
            "conv1_stride": self.conv1_stride,
            "conv1_out_ch": self.conv1_out_ch,
            "pool1_stride": self.pool1_stride,
            "conv2_out_ch": self.conv2_out_ch,
            "conv2_stride": self.conv2_stride, 
            "pool2_stride": self.pool2_stride,
            "conv2_kernel_shape": self.conv2_kernel_shape,
            "pool2_kernel_shape": self.pool2_kernel_shape,
            "conv1_kernel_shape": self.conv1_kernel_shape,
            "pool1_kernel_shape": self.pool1_kernel_shape,
            "components": {k: v.params for k, v in self.layers.items()}
        }


############# LeNet GEMM ################
class LeNet_gemm(object):
    
    def __init__(
        self,
        fc3_out=128,
        fc4_out=84,
        fc5_out=10,
        conv1_pad=0,
        conv2_pad=0,
        conv1_out_ch=6,
        conv2_out_ch=16,
        conv1_stride=1,
        pool1_stride=2,
        conv2_stride=1,
        pool2_stride=2,
        conv1_kernel_shape=(5, 5),
        pool1_kernel_shape=(2, 2),
        conv2_kernel_shape=(5, 5),
        pool2_kernel_shape=(2, 2),
        optimizer="adam",
        init_w="glorot_normal",
        loss=CrossEntropy()
    ):
        self.optimizer = optimizer
        self.init_w = init_w
        self.loss = loss
        self.fc3_out = fc3_out
        self.fc4_out = fc4_out
        self.fc5_out = fc5_out
        self.conv1_pad = conv1_pad
        self.conv2_pad = conv2_pad
        self.conv1_stride = conv1_stride
        self.conv1_out_ch = conv1_out_ch
        self.pool1_stride = pool1_stride
        self.conv2_out_ch = conv2_out_ch
        self.conv2_stride = conv2_stride
        self.pool2_stride = pool2_stride
        self.conv2_kernel_shape = conv2_kernel_shape
        self.pool2_kernel_shape = pool2_kernel_shape
        self.conv1_kernel_shape = conv1_kernel_shape
        self.pool1_kernel_shape = pool1_kernel_shape
        
        self.is_initialized = False
    
    def _set_params(self):
        """
        函数作用：模型初始化
        Conv1 -> Pool1 -> Conv2 -> Pool2 -> Flatten -> FC3 -> FC4 -> FC5 -> Softmax
        """
        self.layers = OrderedDict()
        self.layers["Conv1"] = Conv2D_gemm(
            out_ch=self.conv1_out_ch,
            kernel_shape=self.conv1_kernel_shape,
            pad=self.conv1_pad,
            stride=self.conv1_stride,
            acti_fn="sigmoid",
            optimizer=self.optimizer,
            init_w=self.init_w,
        )
        self.layers["Pool1"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.pool1_stride,
            kernel_shape=self.pool1_kernel_shape,
        )
        self.layers["Conv2"] = Conv2D_gemm(
            out_ch=self.conv1_out_ch,
            kernel_shape=self.conv1_kernel_shape,
            pad=self.conv1_pad,
            stride=self.conv1_stride,
            acti_fn="sigmoid",
            optimizer=self.optimizer,
            init_w=self.init_w,
        )
        self.layers["Pool2"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.pool2_stride,
            kernel_shape=self.pool2_kernel_shape,
        )
        self.layers["Flatten"] = Flatten(optimizer=self.optimizer)
        self.layers["FC3"] = FullyConnected(
            n_out=self.fc3_out,
            acti_fn="sigmoid",
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.layers["FC4"] = FullyConnected(
            n_out=self.fc4_out,
            acti_fn="sigmoid",
            init_w=self.init_w,
            optimizer=self.optimizer
        )
        self.layers["FC5"] = FullyConnected(
            n_out=self.fc5_out,
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
            "fc3_out": self.fc3_out, 
            "fc4_out": self.fc4_out,
            "fc5_out": self.fc5_out,
            "conv1_pad": self.conv1_pad, 
            "conv2_pad": self.conv2_pad, 
            "conv1_stride": self.conv1_stride,
            "conv1_out_ch": self.conv1_out_ch,
            "pool1_stride": self.pool1_stride,
            "conv2_out_ch": self.conv2_out_ch,
            "conv2_stride": self.conv2_stride, 
            "pool2_stride": self.pool2_stride,
            "conv2_kernel_shape": self.conv2_kernel_shape,
            "pool2_kernel_shape": self.pool2_kernel_shape,
            "conv1_kernel_shape": self.conv1_kernel_shape,
            "pool1_kernel_shape": self.pool1_kernel_shape,
            "components": {k: v.params for k, v in self.layers.items()}
        }
