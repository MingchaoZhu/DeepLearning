# Deep Learning

《**深度学习**》是深度学习领域唯一的综合性图书，全称也叫做**深度学习 AI圣经(Deep Learning)**，由三位全球知名专家IanGoodfellow、YoshuaBengio、AaronCourville编著，全书囊括了数学及相关概念的背景知识，包括线性代数、概率论、信息论、数值优化以及机器学习中的相关内容。同时，它还介绍了工业界中实践者用到的深度学习技术，包括深度前馈网络、正则化、优化算法、卷积网络、序列建模和实践方法等，并且调研了诸如自然语言处理、语音识别、计算机视觉、在线推荐系统、生物信息学以及视频游戏方面的应用。最后，深度学习全书还提供了一些研究方向，涵盖的理论主题包括线性因子模型、自编码器、表示学习、结构化概率模型、蒙特卡罗方法、配分函数、近似推断以及深度生成模型，适用于相关专业的大学生或研究生使用。

<img src="https://github.com/MingchaoZhu/DeepLearning/blob/master/docs/cover.jpg" width="200" height="300" alt="深度学习封面" align=center>

可以下载《深度学习》的中文版 [pdf](https://github.com/MingchaoZhu/DeepLearning/releases/download/v0.0.1/DL_cn.pdf) 和英文版 [pdf](https://github.com/MingchaoZhu/DeepLearning/releases/download/v0.0.0/DL_en.pdf) 直接阅读。

对于本项目的工作，你可以直接下载 [深度学习_原理与代码实现.pdf](https://github.com/MingchaoZhu/DeepLearning/releases/download/v1.1.1/default.pdf) (后面会对该书不断更新)

---

《深度学习》可以说是深度学习与人工智能的入门宝典，许多算法爱好者、机器学习培训班、互联网企业的面试，很多都参考这本书。但本书晦涩，加上官方没有提供代码实现，因此某些地方较难理解。本项目**基于数学推导和产生原理重新描述了书中的概念**，并用**Python** (numpy 库为主) 复现了书本内容 ( **源码级代码实现。推导过程和代码实现均放在了下载区的 pdf 文件中**，重要部分的实现代码也放入 **code 文件夹**中 )。

然而我水平有限，但我真诚地希望这项工作可以帮助到更多人学习深度学习算法。我需要大家的建议和帮助。如果你在阅读中遇到有误或解释不清的地方，希望可以汇总你的建议，在 Issues 提出。如果你也想加入这项工作书写中或有其他问题，可以联系我的邮箱。如果你在你的工作或博客中用到了本书，还请可以注明引用链接。

写的过程中参考了较多网上优秀的工作，所有参考资源保存在了`reference.txt`文件中。

# 留言

这份工作就是在写这一本 [深度学习_原理与代码实现.pdf](https://github.com/MingchaoZhu/DeepLearning/releases/download/v1.1.1/default.pdf)。正如你在 pdf 文件中所见到的，《深度学习》涉及到的每一个概念，都会去给它详细的描述、原理层面的推导，以及用代码的实现。代码实现不会调用 Tensorflow、PyTorch、MXNet 等任何深度学习框架，甚至包括 sklearn (pdf 里用到 sklearn 的部分都是用来验证代码无误)，一切代码都是从原理层面实现 (Python 的基础库 NumPy)，并有详细注释，与代码区上方的原理描述区一致，你可以结合原理和代码一起理解。

这份工作的起因是我自身的热爱，但为完成这份工作我需要投入大量的时间精力，一般会写到凌晨两三点。推导、代码、作图都是慢慢打磨的，我会保证这份工作的质量。这份工作会一直更新完，已经上传的章节也会继续补充内容。如果你在阅读过程中遇到有想要描述的概念点或者错误点，请发邮件告知我。

真的很感谢你的认可与推广。最后，请等待下一次更新。

我是 朱明超，我的邮箱是：deityrayleigh@gmail.com

# 更新说明

2020/3/：

```python
1. 修改第五章决策树部分，补充 ID3 和 CART 的原理，代码实现以 CART 为主。
2. 第七章添加 L1 和 L2 正则化最优解的推导 (即 L1稀疏解的原理)。
3. 第七章添加集成学习方法的推导与代码实现，包括 Bagging (随机森林)、Boosting (Adaboost、GBDT、XGBoost)。
4. 第八章添加牛顿法与拟牛顿法 (DFP、BFGS、L-BFGS) 的推导。
5. 第十一章节添加贝叶斯线性回归、高斯过程回归 (GPR) 与贝叶斯优化的推导与代码实现。
```
后面每次的更新内容会统一放在 `update.txt` 文件中。

# 章节目录与文件下载

除了《深度学习》书中的概念点，**本项目也在各章节添加一些补充知识，例如第七章集成学习部分的 随机森林、Adaboost、GBDT、XGBoost 的原理剖析和代码实现等，又或者第十二章对当前一些主流方法的描述**。大的章节目录和 pdf 文件下载链接可以详见下表，而具体 pdf 文件中的实际目录请参考 `contents.txt`。你可以在下面的 pdf 链接中下载对应章节，也可以在 [releases](https://github.com/MingchaoZhu/DeepLearning/releases) 界面直接下载所有文件。

| 中文章节 | 英文章节 | 下载<br />(含推导与代码实现) |
| ------------ | ------------ | ------------ |
| 第一章 前言 | 1 Introduction |  |
| 第二章 线性代数 | 2 Linear Algebra | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/2%20%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0.pdf) |
| 第三章 概率与信息论                 | 3 Probability and Information Theory | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/3%20%E6%A6%82%E7%8E%87%E4%B8%8E%E4%BF%A1%E6%81%AF%E8%AE%BA.pdf) |
| 第四章 数值计算                     | 4 Numerical Computation | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/4%20%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97.pdf) |
| 第五章 机器学习基础                 | 5 Machine Learning Basics | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/5%20%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80.pdf) |
| 第六章 深度前馈网络                 | 6 Deep Feedforward Networks | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/6%20%E6%B7%B1%E5%BA%A6%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C.pdf) |
| 第七章 深度学习中的正则化           | 7 Regularization for Deep Learning | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/7%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E6%AD%A3%E5%88%99%E5%8C%96.pdf) |
| 第八章 深度模型中的优化 | 8 Optimization for Training Deep Models | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/8%20%E6%B7%B1%E5%BA%A6%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E4%BC%98%E5%8C%96.pdf) |
| 第九章 卷积网络 | 9 Convolutional Networks | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/9%20%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C.pdf) |
| 第十章 序列建模：循环和递归网络 | 10 Sequence Modeling: Recurrent and Recursive Nets |  |
| 第十一章 实践方法论                 | 11 Practical Methodology | [pdf](https://github.com/MingchaoZhu/DeepLearning/raw/master/11%20%E5%AE%9E%E8%B7%B5%E6%96%B9%E6%B3%95%E8%AE%BA.pdf) |
| 第十二章 应用 | 12 Applications |  |
| 第十三章 线性因子模型 | 13 Linear Factor Models |  |
| 第十四章 自编码器                   | 14 Autoencoders |  |
| 第十五章 表示学习                   | 15 Representation Learning |  |
| 第十六章 深度学习中的结构化概率模型 | 16 Structured Probabilistic Models for Deep Learning |  |
| 第十七章 蒙特卡罗方法 | 17 Monte Carlo Methods |  |
| 第十八章 直面配分函数 | 18 Confronting the Partition Function |  |
| 第十九章 近似推断                   | 19 Approximate Inference |  |
| 第二十章 深度生成模型 | 20 Deep Generative Models |  |

尚未上传的章节会在后续陆续上传。

# 致谢

感谢对本项目的认可和推广。

+ 专知：https://mp.weixin.qq.com/s/dVD-vKJsMGqnBz2v4O-Q3Q
+ GitHubDaily：https://m.weibo.cn/5722964389/4504392843690487
+ 程序员遇见GitHub：https://mp.weixin.qq.com/s/EzFOnwpkv7mr2TSjPtVG9A
+ 爱可可：https://m.weibo.cn/1402400261/4503389646699745

# 赞助

本项目书写耗费时间精力。如果本项目对你有帮助，可以请作者吃份冰淇淋：

<img src="./docs/pay.jpg" width="200" height="200" alt="支付" align=center>
