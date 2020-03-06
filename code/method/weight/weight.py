from abc import ABC, abstractmethod
import numpy as np
import re


def calc_fan(weight_shape):
    """
    对权重矩阵计算 fan-in 和 fan-out

    参数说明：   
    weight_shape：权重形状
    """
    if len(weight_shape) == 2:  
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError("Unrecognized weight dimension: {}".format(weight_shape))
    return fan_in, fan_out


class random_uniform:
    """
    初始化网络权重 W--- 基于 Uniform(-b, b)

    参数说明：
    weight_shape：权重形状
    """
    def __init__(self, b=1.0):
        self.b = b
        
    def __call__(self, weight_shape):
        return np.random.uniform(-b, b, size=weight_shape)


class random_normal:
    """
    初始化网络权重 W--- 基于 TruncatedNormal(0, std)

    参数说明：   
    weight_shape：权重形状
    std：权重标准差
    """
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, weight_shape):
        return truncated_normal(0, std, weight_shape)

    
# def random_uniform(weight_shape, b=1.0):
#     """
#     初始化网络权重 W--- 基于 Uniform(-b, b)

#     参数说明：
#     weight_shape：权重形状
#     """
#     return np.random.uniform(-b, b, size=weight_shape)


# def random_normal(weight_shape, std=1.0):
#     """
#     初始化网络权重 W--- 基于 TruncatedNormal(0, std)

#     参数说明：   
#     weight_shape：权重形状
#     std：权重标准差
#     """
#     return truncated_normal(0, std, weight_shape)
    

class he_uniform:
    """
    初始化网络权重 W--- 基于 Uniform(-b, b)，其中 b=sqrt(6/fan_in)，常用于 ReLU 激活层

    参数说明：
    weight_shape：权重形状
    """
    def __init__(self):
        pass
    
    def __call__(self, weight_shape):
        fan_in, fan_out = calc_fan(weight_shape)
        b = np.sqrt(6 / fan_in)
        return np.random.uniform(-b, b, size=weight_shape)
    
    
class he_normal:
    """
    初始化网络权重 W--- 基于 TruncatedNormal(0, std)，其中 std=2/fan_in，常用于 ReLU 激活层

    参数说明：   
    weight_shape：权重形状
    """
    def __init__(self):
        pass
    
    def __call__(self, weight_shape):
        fan_in, fan_out = calc_fan(weight_shape)
        std = np.sqrt(2 / fan_in)
        return truncated_normal(0, std, weight_shape)
    
    
    
# def he_uniform(weight_shape):
#     """
#     初始化网络权重 W--- 基于 Uniform(-b, b)，其中 b=sqrt(6/fan_in)，常用于 ReLU 激活层

#     参数说明：
#     weight_shape：权重形状
#     """
#     fan_in, fan_out = calc_fan(weight_shape)
#     b = np.sqrt(6 / fan_in)
#     return np.random.uniform(-b, b, size=weight_shape)
    
    
# def he_normal(weight_shape):
#     """
#     初始化网络权重 W--- 基于 TruncatedNormal(0, std)，其中 std=2/fan_in，常用于 ReLU 激活层

#     参数说明：   
#     weight_shape：权重形状
#     """
#     fan_in, fan_out = calc_fan(weight_shape)
#     std = np.sqrt(2 / fan_in)
#     return truncated_normal(0, std, weight_shape)
    

class glorot_uniform:
    """
    初始化网络权重 W--- 基于 Uniform(-b, b)，其中 b=gain*sqrt(6/(fan_in+fan_out))，
                        常用于 tanh 和 sigmoid 激活层

    参数说明：
    weight_shape：权重形状
    """
    def __init__(self, gain=1.0):
        self.gain = gain
        
    def __call__(self, weight_shape):
        fan_in, fan_out = calc_fan(weight_shape)
        b = self.gain * np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-b, b, size=weight_shape)
    

class glorot_normal:
    """
    初始化网络权重 W--- 基于 TruncatedNormal(0, std)，其中 std=gain^2*2/(fan_in+fan_out)，
                        常用于 tanh 和 sigmoid 激活层

    参数说明：
    weight_shape：权重形状
    """
    def __init__(self, gain=1.0):
        self.gain = gain
        
    def __call__(self, weight_shape):
        fan_in, fan_out = calc_fan(weight_shape)
        std = self.gain * np.sqrt(2 / (fan_in + fan_out))
        return truncated_normal(0, std, weight_shape)
    
    
# def glorot_uniform(weight_shape, gain=1.0):
#     """
#     初始化网络权重 W--- 基于 Uniform(-b, b)，其中 b=gain*sqrt(6/(fan_in+fan_out))，
#                         常用于 tanh 和 sigmoid 激活层

#     参数说明：
#     weight_shape：权重形状
#     """
#     fan_in, fan_out = calc_fan(weight_shape)
#     b = gain * np.sqrt(6 / (fan_in + fan_out))
#     return np.random.uniform(-b, b, size=weight_shape)
    
    
# def glorot_normal(weight_shape, gain=1.0):
#     """
#     初始化网络权重 W--- 基于 TruncatedNormal(0, std)，其中 std=gain^2*2/(fan_in+fan_out)，
#                         常用于 tanh 和 sigmoid 激活层

#     参数说明：
#     weight_shape：权重形状
#     """
#     fan_in, fan_out = calc_fan(weight_shape)
#     std = gain * np.sqrt(2 / (fan_in + fan_out))
#     return truncated_normal(0, std, weight_shape)


def truncated_normal(mean, std, out_shape):
    """
    通过拒绝采样生成截断正态分布

    参数说明：
    mean：正态分布均值
    std：正态分布标准差
    out_shape：矩阵形状
    """
    samples = np.random.normal(loc=mean, scale=std, size=out_shape)
    reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    while any(reject.flatten()):
        resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
        samples[reject] = resamples
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    return samples

    
class WeightInitializer(object):

    def __init__(self, mode="he_normal"):
        """
        mode：权重初始化策略 str型 (default: 'he_normal')
        """
        self.mode = mode
        r = r"([a-zA-Z]*)=([^,)]*)"
        mode_str = self.mode.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, mode_str)])
        
        if "random_uniform" in mode_str:
            self.init_fn = random_uniform(**kwargs)
        elif "random_normal" in mode_str:
            self.init_fn = random_normal(**kwargs)
        elif "he_uniform" in mode_str:
            self.init_fn = he_uniform(**kwargs)
        elif "he_normal" in mode_str:
            self.init_fn = he_normal(**kwargs)
        elif "glorot_uniform" in mode_str:
            self.init_fn = glorot_uniform(**kwargs)
        elif "glorot_normal" in mode_str:
            self.init_fn = glorot_normal(**kwargs)
        else:
            raise ValueError("Unrecognize initialization mode: {}".format(mode_str))
    
    def __call__(self, weight_shape):
        W = self.init_fn(weight_shape)
        return W
