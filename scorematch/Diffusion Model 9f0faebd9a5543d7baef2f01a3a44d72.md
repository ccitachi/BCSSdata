# Diffusion Model

# Score Matching

from “Estimation of Non-Normalized Statistical Models by Score Matching “

$p_\theta(x)=\frac{e^{f_\theta(x)}}{z}$是一个计算出来的函数的概率分布(Probability Density Function，PDF),我们要计算这个归一化的因子$z=\int e^{f_\theta (z)}dx$,保证其是一个概率分布。

**真实数据**的分布为$p_{data}(x)$,**计算出的数据**的分布为$p_\theta(x)$,要拟合使

$$
p_{data}(x)=p_\theta(x)
$$

但是由于要计算$z$这个归一化项，则非常麻烦。

论文的作者提出俩个相似的分布，其log函数的梯度也应该相似，则可以将$p_\theta(x)$的log函数写出：

$$
log(p_\theta(x))=f_\theta(x)-log(z)
$$

$z$是一个常数，则在求梯度的时候直接去除。

则将拟合的公式可以改写为：

$$
\nabla_xlog(p_{data}(x))\simeq\nabla_xlogp_\theta(x)=\nabla_xlog(f_\theta(x))
$$

- Score function: $\nabla_x log(p(x))$
- Score Matching:使用MSE Loss来对俩个Score function 进行计算，找到最优的$\theta$，使Loss最小
- Objective function: 俩个Score function的均方误差，又为Fisher Divergence

$$
\text{Fisher Divergence between two probability distributions }p(x) \text{ and }q(x) = \int (p(x)-\frac{d}{dx}log(p(x)))(q(x)-\frac{d}{dx}log(q(x)))dx
$$

$$
⁍
$$

文中指出 $p_{data}(x)$是不知道的，其对应的Score function： $\nabla_xlog(p_{data}(x))$未知，将上述公式L2范数展开：

$$
L=\frac{1}{2}\int p_{data}[||\nabla_xlogp_{data}(x)||^2-2\nabla_xlogp_{data}^T\nabla_xlogp_{\theta}(x)+||\nabla_xlogp_\theta(x)||^2]dx
$$

我们希望调整 $\theta$来使L更小，其中第一项与$\theta$无关，在计算的时候可以直接丢弃，第三项只与 $p_\theta(x),\theta$有关，我们想消除不好得到的 $p_{data}$,就看第二项能否去除 $p_{data}$。将第二项展开

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled.png)

分布积分的第一项，在 $|x|→\infty,p_{data}(x)->0$,第一项为0，则只有第二项，代入原公式可得：

是求第i个元素的二阶导数，再求和就是Hessian矩阵的迹

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%201.png)

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%202.png)

再使用**蒙特卡洛算法**，得到最终公式：

$$
L=\mathbb{E}[||tr(\nabla_x^2logp_\theta(x)||+\frac{1}{2}||\nabla_xlogp_\theta(x)||^2)|]
$$

**蒙特卡洛算法**

在计算一个数学期望：

$$
\mathbb{E}(f(x))=\int_{\mathbb{R}^n} f(x)p(x)dx
$$

**按照 $p(x)$的概率分布**对 $f(x)$进行**多次采样**，然后再计算均值：

$$
\mathbb{E}(f(x))\simeq \frac{1}{N}\sum_{i=1}^{N}f(x_i)
$$

# **基于梯度估计的生成式模型-宋飏**

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%203.png)

这是个高斯混合模型，俩个高斯分布，颜色代表PDF，那些小箭头代表score。给定PDF，可以求导很快得到score，给定score进行积分得到PDF，所以这俩种是几乎等价。

概率密度函数永远归一化，积分为1，会限制概率密度模型

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%204.png)

score function不会被归一化限制，更加灵活的选择模型结构。

以显式模型能量函数为例：$f_\theta(x)$是一个灵活的函数，但是计算归一化的 $Z_\theta$比较苦难

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%205.png)

score不需要 $Z_\theta$:

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%206.png)

对score estimate 和最开始一样

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%207.png)

这个trace比较难计算，原因是：

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%208.png)

反向计算的量=数据分布的维度，难以应用在大规模数据集上

**Sliced score matching**

高维的向量场进行投影，随机投影得到的标量场沿所有随机方向相同，那得到的高维的向量场也相同

v是一个投影方向

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%209.png)

需要计算 $v^T\nabla_xs_\theta(x)v$ 

第一步相当于多加一个节点

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2010.png)

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2011.png)

只需要一次后向计算 

得到score后，进行采样得到源源不断的样本

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2012.png)

**Langevin MCMC**

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2013.png)

在使用学习得到的score上进行积分，会逐渐坍缩到几个点上，用郎之万采样，加入一定的噪声，可以得到一个对应概率分布的样本。

有一定的问题，在数据概率密度低的区域进行积分函数进行估计比较困难

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2014.png)

数据在密度高的地方，估计的比较准，在密度低的地方，估计的比较差，如果初始化在数据密度低的区域，很难到数据密度高的地方

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2015.png)

方法就是在原图上加噪声，让数据密度更摊开一点，填补一下，有助于密度低的区域也学习的比较好

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2016.png)

加入若干个不同尺度的噪声，因为加入的噪声多，就会影响在中间在离数据密度高的地方，会抹去一些细节信息，

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2017.png)

使用不同的噪声等级进行扰动，

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2018.png)

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2019.png)

若干个不同的噪声等级，每个噪声等级都要学习有一个score-function，这个新的损失函数。

采样的时候先在大噪声上进行采样一个初始化样的，再逐步在小的噪声上进行采样

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2020.png)

如果使用无数种噪声等级，从数据分布，加入无数种噪声扰动，最后到最后一个纯噪声分布：

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2021.png)

要表示这样的一个无穷多的数据分布，可以采用随机过程的方式，t是一个随机变量，x（t）就是在随机变量的控制下，每一个时刻的x数据分布

使用随机过程来描述这个分布：

$$
dx=f(x,t)dt+\sigma(t)dw
$$

第一项是和常微分方程一样，是一些确定的东西，第二项中的$w$是布朗运动白噪声，是随机项，用随机过程去扰动数据分布，不断的增加噪声：

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2022.png)

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2023.png)

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2024.png)

这个随机微分方程可以这样表示，这个扰动项随时间一直增加：

$$
dx=\sigma(t)dw
$$

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2025.png)

逆转噪声过程

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2026.png)

左边是正向过程，右边是对应的反向过程，与score function 有关

如何求解这个反向过程：

需要学习一个函数拟合真实的score function：

$$
s_\theta(x,t)\simeq\nabla_x logp_t(x)
$$

训练：是对无穷个噪声等级的积分 

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2027.png)

反向SDE：

$$
dx=-\sigma^2(t)s_\theta(x,t)dt+\sigma(t)d{w}
$$

Euler-Maruyama数值解法：

无穷小的dt改为$\Delta t$，dw改为均值为0，方差为1的白噪声

$$
x=x-\sigma(t)^2 s_\theta(x,t)\Delta t+\sigma(t)z
$$

# Probability flow ODE: turning the SDE to ODE

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2028.png)

一个SDE都存在一个概率流常微分方程，在t时刻的概率密度函数俩者等同

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2029.png)

常微分方程更加稳定，平滑

求解Probability flow ODE 可以使用**black-box ODE solvers（？？）Neural ODE（？？）**

并且可以求得似然函数：

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2030.png)

# Controllable Generation

条件逆向随机微分方程 

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2031.png)

给定y的x的score function，再使用贝叶斯分开，第一项与原来的一样，第二项可以用学习也可以用经验直接给出

![Untitled](Diffusion%20Model%209f0faebd9a5543d7baef2f01a3a44d72/Untitled%2032.png)