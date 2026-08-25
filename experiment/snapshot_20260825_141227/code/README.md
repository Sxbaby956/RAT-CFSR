# RAT-CFSR

RAT-CFSR 是面向 RML2016.10a 的开放集无线调制识别实现。项目融合：

- S3R 的 STFT、多膨胀率时频纹理编码；
- CFSR 的类投影空间、类自编码器和重构拒识；
- 双视图门控融合、重构排序损失和类别条件误差校准。

## 数据划分

使用 `RML2016.10a_dict.pkl`（11 种调制 × 20 个 SNR(-20~18dB)，每个 (调制, SNR) 有 1000 个 (2,128) 复 IQ 样本）。

- 训练 / 校准 / 测试：在每个 (调制, SNR) 组合内部按 60% / 20% / 20% 随机划分，三个集合都覆盖全部 SNR。
- 每次运行选择一个调制类型作为 unknown，其余 10 个作为 known；unknown 调制只进入测试集。

```bash
python -m rat_cfsr.train --unknown WBFM
python -m rat_cfsr.train --unknown QPSK
```

## 环境

使用 CUDA conda 环境（`RAT-CFSR` 或 `torchsig`）：

```bash
conda activate RAT-CFSR
python -m rat_cfsr.inspect_data --data-root /home/zjut/public/zjm/RML2016.10a
pytest
```

## 常用训练入口

默认运行 RML2016 三未知类单实验（unknown=`AM-DSB AM-SSB WBFM`，`min_snr=0`，输出到 `outputs/rml2016_v8/pruned_prototype_snr0_seed42`）：

```bash
nohup python main.py > logs/rml2016_v8_pruned_prototype_snr0.log 2>&1 &
```

等价于：

```bash
nohup python main.py \
  --single \
  --min-snr 0 \
  --output-dir outputs/rml2016_v8/pruned_prototype_snr0_seed42 \
  > logs/rml2016_v8_pruned_prototype_snr0.log 2>&1 &
```

## 完整训练（开放集矩阵）

一次跑完 11 个 unknown 调制 × 3 个 seed（共 33 个实验）：

```bash
nohup env CUDA_VISIBLE_DEVICES=1 python main.py --matrix \
  --data-root /home/zjut/public/zjm/RML2016.10a \
  --output-dir outputs/rml2016 \
  --batch-size 64 --stage1-epochs 10 --stage2-epochs 20 \
  --modality-dropout 0.3 --open-set-score energy \
  > logs/2016-v1.log 2>&1 &
```

也可以只跑单个实验：

```bash
python -m rat_cfsr.train \
  --data-root /home/zjut/public/zjm/RML2016.10a \
  --output-dir outputs/unknown_WBFM_seed42 \
  --unknown WBFM --seed 42
```

输出目录包含：

- `checkpoint.pt`：模型权重和类别配置；
- `calibrator.json`：每类校准误差和阈值；
- `metrics.json`：AUROC、AUPR、OSCR、TKR、TUR 等；
- `history.json`：两阶段训练曲线；
- `test_predictions.npz`：测试标签、分数和重构误差；
- `split_summary.json`：样本划分统计。

## 重要说明

RML2016.10a 样本已统一为 (2,128)，无需重采样；STFT 分支使用 `n_fft=64`、`hop_length=32`（128 点信号得到 3 帧时频图）。`S3R/` 保留官方源码用于方法追溯，不是新训练入口，新实现入口统一为 `rat_cfsr.train`。
