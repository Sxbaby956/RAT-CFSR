# RAT-CFSR 代码实现与实验执行计划

## 1. 文档目的

本文档完整记录基于 POWDER 数据集、S3R 官方代码和 CFSR 论文所开展的代码实现计划，包括：

- 当前项目中已有内容的审计结果；
- RAT-CFSR 最终 EI 简化架构；
- POWDER 原始 I/Q 数据的解析和严格划分；
- 数据预处理、窗口生成和采样率统一；
- 双视图特征提取、门控融合和类条件重构；
- 损失函数、误差校准和开放集推理；
- 训练、测试、指标和结果文件；
- S3R 官方代码中保留、复用和计划清理的内容；
- 已完成代码、尚未完成验证以及后续执行顺序；
- 单元测试、冒烟测试、完整实验和最终验收标准。

本文档既是研发计划，也可以作为后续论文“系统实现”和“实验设置”部分的技术依据。

---

## 2. 当前项目状态

### 2.1 已经存在的主要资源

项目根目录：

```text
C:\Users\thinkpad\Desktop\zjm
```

主要内容包括：

```text
zjm/
├── GlobecomPOWDER/                         # POWDER 原始 I/Q 数据
├── S3R/                                    # S3R 官方实现及旧实验输出
├── rat_cfsr/                               # 已创建的新架构代码包
├── tests/                                  # 已创建的单元测试
├── README.md                               # 新项目使用说明
├── pyproject.toml                          # Python 项目和依赖配置
├── requirements.txt                        # 依赖列表
├── .gitignore                              # 忽略环境、输出和权重
├── 面向无线信号体制开放集识别的融合架构方案.md
├── 融合架构与两篇论文的继承关系说明.md
├── RAT-CFSR_整体架构图.svg
└── RAT-CFSR_整体架构图.png
```

### 2.2 已完成的代码文件

目前已经撰写以下首版代码：

```text
rat_cfsr/
├── __init__.py
├── data.py
├── model.py
├── losses.py
├── calibration.py
├── metrics.py
├── train.py
└── inspect_data.py

tests/
├── test_data.py
├── test_model.py
└── test_calibration.py
```

这些代码已经写入项目，但由于依赖安装过程被中断，尚未完成 PyTorch 环境下的单元测试和端到端验证。

### 2.3 尚未完成的工作

当前仍需完成：

1. 完整安装隔离 Python 环境中的 PyTorch、scikit-learn 和 pytest；
2. 运行三个单元测试；
3. 使用真实 POWDER 数据执行一次 `--dry-run`；
4. 使用极小数据配置完成一轮 Stage 1 和一轮 Stage 2 训练；
5. 验证类别条件校准、未知拒识和指标输出；
6. 根据测试结果修复可能存在的张量维度、速度或数值稳定性问题；
7. 清理 S3R 目录中确认无用的历史输出；
8. 运行三种未知类别轮换的正式实验。

### 2.4 被中断的环境安装状态

项目中已经创建：

```text
.venv/
```

但环境安装被中断，当前 `.venv` 中尚未成功安装：

- PyTorch；
- scikit-learn；
- pytest。

被中断后仍在后台运行的依赖安装进程已经停止，避免继续占用网络和磁盘。后续应重新执行依赖安装，不应假设当前 `.venv` 已可直接使用。

---

## 3. POWDER 数据审计结果

### 3.1 信号体制

数据目录中包含三种协议前缀：

- `4G`：LTE；
- `5G`：5G NR；
- `WiFi`：IEEE 802.11a。

### 3.2 采集维度

每种体制包括：

- 2 个采集日：Day 1、Day 2；
- 4 个发射基站：`bes`、`browning`、`honors`、`meb`；
- 每个组合 5 个录音集合：s1–s5。

理论录音数量：

$$3\times2\times4\times5=120$$

实际项目文件结构与这一数量一致。每组录音由一对文件组成：

```text
*.bin     # 原始复数 I/Q
*.json    # SigMF 风格元数据
```

### 3.3 数据类型

JSON 中的：

```json
"core:datatype": "cf32"
```

表示每个采样点是 32 位实部和 32 位虚部组成的复数，即 NumPy：

```python
np.complex64
```

每个复数采样占 8 字节。

例如：

```text
4G_Day_1_bes_s1.bin       42,400,000 bytes
```

对应：

$$42,400,000/8=5,300,000$$

个复数采样，与 JSON 的 `core:sample_count` 一致。

### 3.4 采样率差异

LTE 和 5G NR：

```text
7.69 MS/s
```

Wi-Fi：

```text
5 MS/s
```

这是当前实验最大的潜在数据捷径。如果不同类别直接生成相同采样点数的窗口，那么它们对应的真实时间长度不同；如果直接使用相同真实时间截窗但不重采样，那么网络又能从张量长度或频谱分辨率推断类别。

因此，新数据管线采用：

1. 先按照相同真实持续时间从不同采样率录音中截取；
2. 再将窗口统一重采样为相同点数；
3. 网络只看到统一形状的输入。

### 3.5 中心频率

样例元数据中三类信号中心频率均为：

```text
2.685 GHz
```

新代码不把中心频率作为模型输入，避免引入非波形本质特征。

---

## 4. 严格数据划分

### 4.1 默认划分

新代码默认使用：

| 集合 | 数据范围 | 是否包含未知体制 |
|---|---|---|
| Train | Day 1，s1–s4 | 否 |
| Calibration | Day 1，s5 | 否 |
| Test | Day 2，s1–s5 | 是 |

### 4.2 划分顺序

必须先按完整录音划分，再从录音中生成窗口：

```text
完整录音
   ↓
按 day / set 划分 Train、Calibration、Test
   ↓
各集合内部独立生成窗口
```

严禁：

```text
完整录音
   ↓
先生成大量重叠窗口
   ↓
随机划分 Train/Test
```

后一种方式会把同一条录音中相邻且高度相关的窗口同时放进训练集和测试集，造成严重数据泄漏。

### 4.3 三种开放集轮换

正式实验分别执行：

| 实验 | 已知体制 | 未知体制 |
|---|---|---|
| A | LTE、Wi-Fi | 5G NR |
| B | 5G NR、Wi-Fi | LTE |
| C | 5G NR、LTE | Wi-Fi |

训练和校准只加载已知体制；测试加载全部三种体制，并将被留出的体制标签转换为 `-1`。

---

## 5. 数据管线设计

### 5.1 元数据发现

`rat_cfsr.data.discover_recordings()` 负责：

1. 遍历 `GlobecomPOWDER/*.json`；
2. 解析文件名中的协议、day、基站和 set；
3. 找到对应 `.bin`；
4. 校验 `core:datatype == cf32`；
5. 比较文件字节数和元数据 sample count；
6. 校验文件名协议与 JSON 协议一致；
7. 校验发射基站名称；
8. 返回统一的 `Recording` 记录。

### 5.2 内存映射读取

数据文件较大，不应一次性加载到内存。代码采用：

```python
np.memmap(path, dtype="<c8", mode="r")
```

特点：

- 只读取当前窗口需要的磁盘范围；
- 不复制完整录音；
- 适合反复随机访问窗口；
- 每个 DataLoader worker 使用自己的 memmap 缓存。

### 5.3 固定真实时间窗口

假设窗口时长为 $T_w$ 毫秒，原始窗口点数为：

$$L_{raw}=\operatorname{round}(f_sT_w/1000)$$

例如 $T_w=1$ ms：

- LTE/NR：约 7690 点；
- Wi-Fi：5000 点。

两者覆盖相同实际时间。

### 5.4 固定网络输入长度

每个原始窗口归一化后被插值为统一长度，例如：

```text
8192 complex samples
```

最终张量：

```text
[2, 8192]
```

两个通道分别是 I 和 Q。

首版使用线性复数插值：

- 实部分别插值；
- 虚部分别插值；
- 重新组合成 complex64。

后续可以对比 `scipy.signal.resample_poly`。如果线性插值造成高频信息衰减或引入可识别的插值模式，应改用多相滤波重采样。

### 5.5 信号归一化

每个窗口执行：

$$x'=\frac{x-\operatorname{mean}(x)}{\sqrt{\operatorname{mean}(|x|^2)}+\epsilon}$$

该步骤：

- 去除直流分量；
- 统一 RMS 功率；
- 降低距离、链路衰落和接收增益成为类别捷径的风险。

### 5.6 训练增强

首版包含：

1. 随机全局相位旋转；
2. 小范围归一化频移；
3. 随机 AWGN；
4. 增强后再次 RMS 归一化。

默认随机频移范围：

```text
±0.02 cycles/sample
```

默认 AWGN：

- 触发概率 0.5；
- SNR 15–35 dB。

校准和测试阶段不进行随机增强。

### 5.7 窗口数量控制

完整录音可以生成大量窗口。为了控制首轮实验规模，使用：

```text
--max-windows-per-recording 256
```

代码在整条录音的可用窗口位置上均匀抽取，避免全部窗口集中在录音开头。

正式实验可以提高到 512 或使用全部非重叠窗口，但应先完成小规模稳定性验证。

---

## 6. RAT-CFSR 网络结构

### 6.1 总体数据流

```text
原始 I/Q [B,2,L]
     │
     ├── IQ Encoder ───────────── z_iq [B,D]
     │
     └── STFT → Multi-scale TE ─ z_tf [B,D]
                                      │
                             Gated Fusion
                                      │
                                  z [B,D]
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                    Class Head              K Class Branches
                         │                         │
                       logits       projector_k → AE_k → e_k
```

### 6.2 I/Q 时域编码器

输入：

```text
[B, 2, L]
```

结构：

1. 一维卷积 stem；
2. BatchNorm；
3. ReLU；
4. MaxPool；
5. 多个一维残差块；
6. Adaptive Average Pool；
7. Linear + LayerNorm + GELU。

主要通道变化：

```text
2 → 32 → 64 → 128 → 128
```

输出：

```text
z_iq ∈ R^128
```

该分支继承 CFSR 使用原始 I/Q 和一维残差骨干的思想。

### 6.3 STFT 前端

模型内部在 GPU/CPU 上直接执行 `torch.stft`，而不是把谱图提前保存成大量图片。

默认：

```text
n_fft      = 256
hop_length = 128
window     = Hann
center     = False
```

复数谱转换为：

$$X=\log(1+|\operatorname{STFT}(x)|)$$

每个样本单独标准化：

$$\hat X=\frac{X-\mu_X}{\sigma_X+\epsilon}$$

### 6.4 多尺度时频编码器

借鉴 S3R 的 Texture Extractor，设置三个二维卷积分支：

```text
dilation = 1
dilation = 3
dilation = 5
```

每个分支包括：

- Conv2d；
- BatchNorm2d；
- ReLU；
- MaxPool2d；
- 第二层同膨胀率卷积；
- Adaptive Average Pool。

每个分支输出 32 维，三个分支连接后得到 96 维，再映射到：

```text
z_tf ∈ R^128
```

### 6.5 自适应门控融合

连接两个分支：

$$u=[z_{iq};z_{tf}]$$

门控 MLP 输出两个 logits，并经 softmax：

$$[g_{iq},g_{tf}]=\operatorname{softmax}(\operatorname{MLP}(u))$$

融合：

$$z=\operatorname{LayerNorm}(g_{iq}z_{iq}+g_{tf}z_{tf})$$

训练阶段启用模态屏蔽：

- 一定概率屏蔽 I/Q 分支；
- 一定概率屏蔽时频分支；
- 永远不会同时屏蔽两个分支。

目的：防止门控永久退化为只使用一个分支。

### 6.6 已知类辅助分类头

```text
Linear(128, K)
```

用于计算已知类别交叉熵。该分类头主要服务训练阶段；最终开放集决策由校准后的重构分数完成。

### 6.7 类投影器

融合语义先做 L2 归一化：

$$\bar z=\frac{z}{\|z\|_2+\epsilon}$$

每个已知类建立独立投影器：

```text
128 → 128 → 64 → tanh
```

输出：

$$h_k=p_k(\bar z)$$

`tanh` 把类空间限制在有界范围，防止模型仅通过无限增大错误类特征满足 margin。

### 6.8 瓶颈类自编码器

每类一个独立 AE：

```text
64 → 32 → 16 → 32 → 64
```

编码器与解码器之间没有跳跃连接。

训练时可对 AE 输入添加轻微高斯噪声：

```text
std = 0.01
```

该设计降低 AE 退化成恒等映射的风险。

### 6.9 重构误差

每类支路输出：

$$e_k(x)=\frac{1}{64}\|h_k-AE_k(h_k)\|_1$$

模型一次前向输出全部：

```text
[B, K]
```

重构误差矩阵。

---

## 7. 损失函数

总损失：

$$L=L_{ce}+\lambda_{rec}L_{rec}+\lambda_{margin}L_{margin}$$

### 7.1 已知类交叉熵

$$L_{ce}=-\frac1M\sum_i\log p(y_i|x_i)$$

作用：

- 保证融合语义具备闭集分类能力；
- 避免只优化重构造成语义缺乏判别性。

### 7.2 正确类重构损失

$$L_{rec}=\frac1M\sum_i e_{y_i}(x_i)$$

只要求样本在正确类别支路中被低误差重构。

### 7.3 重构排序损失

$$L_{margin}=\frac1M\sum_i\max(0,m+e_{y_i}-\min_{k\ne y_i}e_k)$$

其中：

- $e_{y_i}$：正确类误差；
- $\min_{k\ne y_i}e_k$：最容易混淆的错误类误差；
- $m$：要求保持的误差间隔。

该损失把“其他已知类相对于当前类就是伪未知”的 CFSR 思想转化为直接服务推理规则的误差排序约束。

### 7.4 默认权重

```text
lambda_rec    = 1.0
lambda_margin = 0.5
margin        = 0.2
```

这些是初始工程值，正式论文必须使用已知验证/校准数据选择，不能根据测试未知类结果调参。

---

## 8. 两阶段训练

### 8.1 Stage 1：双视图语义预训练

参与训练：

- I/Q Encoder；
- STFT Encoder；
- Gated Fusion；
- Classifier。

损失：

```text
L = L_ce
```

目的：先让双视图语义能够稳定区分已知体制。

默认：

```text
epochs = 10
lr     = 3e-4
```

使用 Calibration 集的已知分类准确率选择最佳 Stage 1 权重。

### 8.2 Stage 2：类空间重构联合微调

加入：

- K 个类投影器；
- K 个瓶颈 AE；
- $L_{rec}$；
- $L_{margin}$。

优化器使用两组学习率：

```text
双视图骨干和融合模块：3e-5
投影器和 AE：          1e-4
```

默认 20 epoch。

每个 epoch 记录：

- 总损失；
- 分类损失；
- 正确类重构损失；
- margin 损失；
- 正确类平均误差；
- 最难错误类平均误差；
- Calibration 闭集准确率；
- 两个门控权重均值；
- 训练耗时。

### 8.3 需要重点观察的异常

#### 所有 AE 误差同时很低

可能原因：

- AE 容量过大；
- 投影空间塌缩；
- AE 学成恒等映射。

处理：

- 缩小 bottleneck；
- 增大输入噪声；
- 检查正确和错误类误差间隔；
- 增大 margin 权重。

#### 门控权重长期接近 `[1,0]` 或 `[0,1]`

可能原因：

- 某分支明显更容易优化；
- 时频分支尺度异常；
- 模态屏蔽概率太低。

处理：

- 检查两个语义的均值和方差；
- 提高 modality dropout；
- 做单分支消融确认是否确实不需要另一分支。

---

## 9. 类别条件误差校准

### 9.1 为什么需要校准

不同类投影器和 AE 的原始重构误差尺度可能不同。

直接执行：

$$\arg\min_k e_k$$

可能长期偏向正常误差天然较小的某个类 AE。

### 9.2 校准数据

校准只使用：

```text
Day 1, set 5, known protocols only
```

它不参与梯度更新，也不包含测试未知类。

### 9.3 每类参考分布

对类别 $k$：

$$E_k^{cal}=\{e_k(x_i):y_i=k\}$$

将误差升序保存。

### 9.4 经验 CDF 分数

测试误差转换为：

$$s_k(x)=F_k(e_k(x))$$

代码通过 `np.searchsorted` 计算误差在该类参考分布中的相对秩。

解释：

- $s_k$ 小：误差低于该类大部分已知样本，较符合类 $k$；
- $s_k$ 接近 1：位于该类正常误差分布尾部，较不像类 $k$。

### 9.5 候选类别

$$k^*=\arg\min_k s_k(x)$$

### 9.6 未知拒识

$$\hat y=
\begin{cases}
k^*,&s_{k^*}\le\tau_{k^*}\\
-1,&s_{k^*}>\tau_{k^*}
\end{cases}$$

默认阈值分位数：

```text
0.95
```

正式实验应比较：

```text
0.90 / 0.95 / 0.99
```

### 9.7 保存内容

`calibrator.json` 保存：

- threshold quantile；
- 每类阈值；
- 每类完整排序校准误差。

模型权重和校准器分开保存，避免部署时误把训练误差当成校准分布。

---

## 10. 开放集评价指标

### 10.1 Known Accuracy

已知测试样本中被正确分类为准确已知类别的比例。

### 10.2 True Known Rate（TKR）

$$TKR=\frac{\text{被接受的已知样本}}{\text{全部已知样本}}$$

### 10.3 True Unknown Rate（TUR）

$$TUR=\frac{\text{被拒识的未知样本}}{\text{全部未知样本}}$$

### 10.4 Known Precision（KP）

$$KP=\frac{\text{正确分类的已知样本}}{\text{全部被接受样本}}$$

### 10.5 Unknown Precision

$$UP_{reject}=\frac{\text{正确拒识的未知样本}}{\text{全部被拒识样本}}$$

这里仅表示拒识精度，不等同于 S3R 中包含未知聚类效果的 UP。

### 10.6 AUROC 和 AUPR

连续未知分数使用：

$$S_{unk}(x)=\min_k s_k(x)$$

分数越高越倾向未知。

### 10.7 OSCR

代码实现 Correct Classification Rate 与未知 False Positive Rate 曲线面积，联合衡量：

- 已知样本是否被正确细分；
- 未知样本是否被拒识。

### 10.8 Macro-F1

将 `-1` 作为 unknown 标签，与已知标签共同计算开放集 Macro-F1。

---

## 11. 代码模块说明

### 11.1 `rat_cfsr/data.py`

负责：

- 解析 POWDER 文件名和 JSON；
- 校验 cf32 和 sample count；
- 生成 Train/Calibration/Test 录音划分；
- 根据真实时间生成窗口；
- memmap 读取；
- 重采样；
- RMS 归一化；
- I/Q 数据增强；
- 返回 PyTorch Dataset。

### 11.2 `rat_cfsr/model.py`

包含：

- `ResidualBlock1D`；
- `IQEncoder`；
- `DilatedSpectrogramBranch`；
- `SpectrogramEncoder`；
- `GatedFusion`；
- `ClassProjector`；
- `BottleneckAutoencoder`；
- `RATCFSR`。

### 11.3 `rat_cfsr/losses.py`

实现：

- CE；
- 正确类重构；
- 最难负类排序；
- Stage 1 classification-only；
- Stage 2 完整损失。

### 11.4 `rat_cfsr/calibration.py`

实现：

- 每类匹配误差收集；
- 经验 CDF 转换；
- 每类阈值；
- 已知/未知预测；
- JSON 保存和加载。

### 11.5 `rat_cfsr/metrics.py`

实现：

- Known Accuracy；
- TKR；
- TUR；
- KP；
- Unknown Precision；
- Macro-F1；
- AUROC；
- AUPR；
- OSCR。

### 11.6 `rat_cfsr/train.py`

实现完整命令行流程：

1. 设置随机种子；
2. 选择 CPU/GPU；
3. 构建数据集；
4. 打印划分统计；
5. 构建模型；
6. dry-run；
7. Stage 1；
8. Stage 2；
9. 校准；
10. 测试；
11. 保存权重、指标和预测。

### 11.7 `rat_cfsr/inspect_data.py`

不训练模型，只输出：

- 数据总录音数；
- 每类录音数；
- 采样率；
- 基站；
- 各 split 录音数；
- 各 split 窗口数。

---

## 12. 命令行参数

### 12.1 数据相关

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data-root` | `GlobecomPOWDER` | 数据目录 |
| `--unknown` | `5G` | 本次留出的未知体制 |
| `--window-ms` | `1.0` | 每个窗口真实时长 |
| `--stride-fraction` | `1.0` | 步长相对窗口长度 |
| `--num-iq-samples` | `8192` | 重采样后输入点数 |
| `--max-windows-per-recording` | `256` | 每条录音最多窗口数 |

### 12.2 训练相关

| 参数 | 默认值 |
|---|---:|
| `--batch-size` | 64 |
| `--workers` | 0 |
| `--stage1-epochs` | 10 |
| `--stage2-epochs` | 20 |
| `--stage1-lr` | 3e-4 |
| `--backbone-lr` | 3e-5 |
| `--class-module-lr` | 1e-4 |
| `--weight-decay` | 1e-4 |

Windows 上首轮测试使用 `workers=0`，确认稳定后再增加 worker。

### 12.3 模型相关

| 参数 | 默认值 |
|---|---:|
| `--semantic-dim` | 128 |
| `--projection-dim` | 64 |
| `--bottleneck-dim` | 16 |
| `--n-fft` | 256 |
| `--hop-length` | 128 |
| `--modality-dropout` | 0.1 |
| `--ae-noise-std` | 0.01 |

### 12.4 开放集相关

| 参数 | 默认值 |
|---|---:|
| `--reconstruction-weight` | 1.0 |
| `--margin-weight` | 0.5 |
| `--margin` | 0.2 |
| `--threshold-quantile` | 0.95 |

### 12.5 运行相关

| 参数 | 说明 |
|---|---|
| `--output-dir` | 保存权重、校准器和结果 |
| `--seed` | 随机种子 |
| `--device` | `auto/cpu/cuda/cuda:0` |
| `--dry-run` | 仅执行数据和一次前向，不训练 |

---

## 13. 结果文件

每次正式训练输出：

```text
outputs/unknown_5g_seed42/
├── checkpoint.pt
├── calibrator.json
├── config.json
├── split_summary.json
├── history.json
├── metrics.json
└── test_predictions.npz
```

### 13.1 `checkpoint.pt`

包含：

- 模型 state dict；
- 已知协议顺序；
- 未知协议；
- 模型结构参数。

### 13.2 `calibrator.json`

包含：

- 每类校准误差；
- 阈值；
- threshold quantile。

### 13.3 `history.json`

包含每个 epoch 的：

- 各项损失；
- Calibration 准确率；
- 门控均值；
- 训练耗时。

### 13.4 `test_predictions.npz`

包含：

- true labels；
- predicted labels；
- candidate labels；
- candidate scores；
- class scores；
- reconstruction errors；
- logits。

该文件用于后续绘制：

- ROC；
- PR；
- OSCR；
- 重构误差分布；
- 每类阈值图；
- 混淆矩阵。

---

## 14. 单元测试计划

### 14.1 数据测试

文件：

```text
tests/test_data.py
```

验证：

- 能读取临时 cf32 文件；
- 能解析 JSON；
- 能按真实时间生成窗口；
- 重采样后张量为固定形状；
- 标签正确；
- 数值有限。

### 14.2 模型测试

文件：

```text
tests/test_model.py
```

验证：

- logits 形状；
- 重构误差形状；
- 门控权重和为 1；
- 总损失有限；
- 反向传播成功。

### 14.3 校准测试

文件：

```text
tests/test_calibration.py
```

构造：

- 类 0 低误差样本；
- 类 1 低误差样本；
- 一个两个类误差都极大的未知样本。

验证校准器输出：

```text
[0, 1, -1]
```

---

## 15. 实际数据冒烟测试

### 15.1 第一步：数据检查

```powershell
.\.venv\Scripts\python.exe -m rat_cfsr.inspect_data `
  --data-root GlobecomPOWDER `
  --unknown 5G `
  --window-ms 1.0 `
  --max-windows-per-recording 4
```

预期：

- 总录音 120；
- Train 只含两个已知体制；
- Calibration 只含两个已知体制；
- Test 含全部三种体制。

### 15.2 第二步：前向 dry-run

```powershell
.\.venv\Scripts\python.exe -m rat_cfsr.train `
  --data-root GlobecomPOWDER `
  --unknown 5G `
  --max-windows-per-recording 4 `
  --batch-size 4 `
  --num-iq-samples 2048 `
  --n-fft 128 `
  --hop-length 64 `
  --dry-run
```

应输出：

- 输入形状；
- logits 形状 `[B,2]`；
- errors 形状 `[B,2]`；
- spectrogram 形状；
- 总参数量。

### 15.3 第三步：一轮端到端训练

```powershell
.\.venv\Scripts\python.exe -m rat_cfsr.train `
  --data-root GlobecomPOWDER `
  --output-dir outputs\smoke `
  --unknown 5G `
  --max-windows-per-recording 2 `
  --batch-size 4 `
  --num-iq-samples 1024 `
  --n-fft 64 `
  --hop-length 32 `
  --stage1-epochs 1 `
  --stage2-epochs 1 `
  --device cpu
```

验收：

- 训练不报错；
- loss 可反传；
- 校准每类至少有样本；
- 能生成所有结果文件；
- `metrics.json` 数值有效；
- unknown 标签为 `-1`。

---

## 16. 正式实验计划

### 16.1 三种未知体制

对：

```text
unknown = 5G
unknown = 4G
unknown = WiFi
```

分别训练。

### 16.2 随机种子

最低使用：

```text
42 / 123 / 2026
```

总训练次数：

$$3\text{种未知}\times3\text{个种子}=9$$

### 16.3 正式命令示例

```powershell
.\.venv\Scripts\python.exe -m rat_cfsr.train `
  --data-root GlobecomPOWDER `
  --output-dir outputs\unknown_5G_seed42 `
  --unknown 5G `
  --window-ms 1.0 `
  --num-iq-samples 8192 `
  --max-windows-per-recording 256 `
  --batch-size 64 `
  --stage1-epochs 10 `
  --stage2-epochs 20 `
  --seed 42
```

### 16.4 汇总结果

最终对每种未知体制报告：

- 3 个种子均值；
- 标准差；
- AUROC；
- AUPR；
- OSCR；
- Known Accuracy；
- TKR；
- TUR；
- KP；
- 参数量；
- 推理时间。

---

## 17. 消融实验计划

### 17.1 表示消融

1. 仅 I/Q 分支；
2. 仅 STFT 分支；
3. 双分支直接平均；
4. 双分支门控融合。

### 17.2 重构消融

1. 仅分类头 + MSP；
2. 分类头 + 类 AE；
3. 分类头 + 类 AE + margin；
4. 完整模型 + 校准。

### 17.3 校准消融

1. 原始误差：$\arg\min e_k$；
2. Median/IQR 标准化；
3. 经验 CDF 校准。

### 17.4 AE 消融

1. 无瓶颈或较宽 AE；
2. 64→32→64；
3. 64→16→64；
4. 64→8→64。

### 17.5 阈值消融

```text
0.90 / 0.95 / 0.99
```

### 17.6 窗口长度消融

至少比较两种真实时长。具体值需要根据显存和数据结构决定，可先从：

```text
0.5 ms / 1.0 ms / 2.0 ms
```

开始。

### 17.7 数据泄漏对照

可增加一项仅用于说明问题的实验：

- 窗口随机划分；
- session/day 严格划分。

论文主结果只能采用严格划分。

---

## 18. S3R 官方代码复用策略

### 18.1 保留并参考的内容

建议保留：

```text
S3R/README.md
S3R/train.py
S3R/contLoss.py
S3R/test_stage_1.py
S3R/test_stage_2.py
S3R/raw2tfs/plt.py
S3R/experiment_groups/
S3R/run_test.sh
```

保留理由：

- 方法可追溯；
- 可对照多膨胀率 TE；
- 可对照中心/间隔损失；
- 可对照 S3R 原始阈值和指标实现；
- 后续论文写作时便于核对。

### 18.2 已复用的思想

从 S3R 复用：

- STFT 时频谱；
- 单样本归一化；
- 多尺度膨胀卷积；
- 语义类间分离思想。

没有直接复制原代码网络，因为其：

- 输入是 512×512 无人机谱图；
- 数据加载含绝对 Linux 路径；
- PE 使用时间/频率绝对位置注意力；
- 分类阶段使用马氏距离和 3σ；
- Stage 2 针对未知无人机聚类；
- 与 POWDER 的原始 I/Q、三类 RAT 任务不匹配。

### 18.3 计划删除的生成物

完成新代码测试并再次确认路径后，可以删除以下旧运行产物：

```text
S3R/__pycache__/
S3R/*.log
S3R/centers.npy
S3R/dist_matrix.npy
S3R/label_hat.npy
S3R/test_X.npy
S3R/test_X_expand.npy
S3R/test_Y.npy
S3R/theta.npy
S3R/model/
S3R/semantic/
```

这些文件是旧无人机实验的：

- 缓存；
- 训练日志；
- 模型权重；
- 测试语义；
- 阈值和距离矩阵。

它们不被新 `rat_cfsr/` 代码引用。

### 18.4 当前删除状态

截至本文档生成时，上述 S3R 生成物尚未删除。原因是环境安装和测试被中断，尚未完成新代码端到端验证。应在测试通过后再执行清理，避免在新实现尚未确认时过早移除可参考结果。

### 18.5 绝对不能删除的内容

未经明确确认，不删除：

- `GlobecomPOWDER/` 中任何 `.bin` 或 `.json`；
- 两篇原论文 PDF；
- S3R 源码和 README；
- 当前架构文档和图；
- 新 `rat_cfsr/` 源码；
- 新测试文件。

---

## 19. 环境安装计划

### 19.1 CPU 验证环境

建议先安装 CPU 版 PyTorch，完成代码正确性验证：

```powershell
uv venv --python 3.12 .venv

uv pip install `
  --python .venv\Scripts\python.exe `
  torch `
  --index-url https://download.pytorch.org/whl/cpu

uv pip install `
  --python .venv\Scripts\python.exe `
  -e ".[test]"
```

### 19.2 GPU 正式环境

CPU 冒烟测试通过后，根据本机 CUDA 驱动安装匹配的 PyTorch GPU 版本。

正式训练前运行：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 19.3 依赖

最低依赖：

- Python 3.10–3.12；
- NumPy；
- PyTorch；
- scikit-learn；
- pytest。

---

## 20. 已知风险与处理方案

### 20.1 线性重采样留下类别痕迹

风险：Wi-Fi 和 LTE/NR 的不同上采样比例可能产生不同插值纹理。

处理：

- 增加随机二次重采样；
- 对比多相滤波；
- 训练一个只识别原始采样率的捷径基线；
- 做频谱检查。

### 20.2 三类数据导致开放度有限

POWDER 只有三种体制，每次只能两类已知、一类未知。

处理：

- 三种未知轮换全部执行；
- 多随机种子；
- 跨 day 和跨基站；
- 后续增加 Ghent LTE/Wi-Fi/DVB-T 数据作为第二数据集。

### 20.3 AE 对未知信号也能重构

处理：

- 低维瓶颈；
- 有界投影；
- 无 skip connection；
- AE 输入噪声；
- margin 排序；
- 监控错误类误差。

### 20.4 短窗口不能区分相似 OFDM

处理：

- 增加窗口长度敏感性；
- 观察 LTE/NR/Wi-Fi 混淆矩阵；
- 如果双分支仍无法区分，再考虑加入轻量周期结构分支，而不是一开始增加复杂 Transformer。

### 20.5 Calibration 阈值对域变化敏感

Day 1 的校准分布可能不能覆盖 Day 2 的已知类变化。

处理：

- 单独报告 shifted-known 误拒率；
- 比较 0.90/0.95/0.99；
- 使用更强的频移、AWGN 和相位增强；
- 后续研究域鲁棒校准。

### 20.6 CPU STFT 速度慢

处理：

- 首轮降低输入长度和窗口数；
- 正式训练使用 GPU；
- 必要时预缓存经过统一重采样的 I/Q，而不缓存图片谱图；
- 评估 DataLoader workers。

---

## 21. 分阶段里程碑

### 里程碑 1：代码可导入

验收：

- `import rat_cfsr` 成功；
- 所有模块无语法错误；
- `python -m rat_cfsr.inspect_data` 成功。

### 里程碑 2：单元测试通过

验收：

```text
3 passed
```

并且：

- 数据形状正确；
- 模型可反传；
- 校准器能拒绝构造未知样本。

### 里程碑 3：真实数据 dry-run

验收：

- 成功读取 POWDER memmap；
- 不加载完整录音到内存；
- I/Q、STFT、logits、errors 形状正确；
- 无 NaN/Inf。

### 里程碑 4：端到端冒烟训练

验收：

- Stage 1 成功；
- Stage 2 成功；
- 校准成功；
- 测试成功；
- 所有输出文件生成。

### 里程碑 5：单个未知体制正式训练

先完成：

```text
unknown = 5G, seed = 42
```

检查：

- 训练曲线；
- 正确/错误类误差；
- 门控权重；
- 已知和未知分数直方图；
- AUROC、OSCR、TUR。

### 里程碑 6：三未知×三种子

生成论文主结果表。

### 里程碑 7：基线与消融

完成 EI 论文实验矩阵。

### 里程碑 8：清理 S3R 生成物

仅在新流程确认稳定后执行。

---

## 22. 最终验收标准

代码层面：

- 所有测试通过；
- 训练命令可复现；
- 不包含硬编码绝对路径；
- 不把未知样本用于训练或校准；
- 每次输出完整配置和划分摘要；
- 数据读取不会一次性载入全部 POWDER。

实验层面：

- 三种未知体制全部测试；
- 至少三个随机种子；
- 严格 Day 1/Day 2 隔离；
- 报告已知域迁移误拒率；
- 报告 AUROC、AUPR、OSCR、TKR、TUR；
- 包含单分支、门控、重构和校准消融。

论文层面：

- 明确 S3R 与 CFSR 的继承关系；
- 不把简单 STFT 说成独立创新；
- 将门控融合、重构排序和类别条件校准作为主要改造；
- 公开说明 POWDER 采样率差异及处理方式；
- 公开说明录音级划分，避免数据泄漏质疑。

---

## 23. 推荐的下一步执行顺序

下一次继续开发时，严格按照以下顺序：

1. 重新安装 `.venv` 依赖；
2. 运行 `pytest`；
3. 修复所有单元测试问题；
4. 运行 `inspect_data`；
5. 运行真实 POWDER `--dry-run`；
6. 检查 STFT 和模型张量形状；
7. 运行一轮 Stage 1 + 一轮 Stage 2；
8. 检查 `calibrator.json` 和 `metrics.json`；
9. 修复数值和性能问题；
10. 执行一个完整未知体制实验；
11. 确认新流程稳定；
12. 清理 S3R 旧生成物；
13. 执行三未知×三种子；
14. 开始基线和消融实验。

该顺序能最大限度避免在环境、数据或模型基础问题尚未解决时直接投入长时间正式训练。
