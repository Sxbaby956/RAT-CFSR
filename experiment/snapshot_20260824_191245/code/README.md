# RAT-CFSR

RAT-CFSR 是面向 5G NR、LTE 和 IEEE 802.11a Wi-Fi 的开放集无线体制识别实现。项目融合：

- S3R 的 STFT、多膨胀率时频纹理编码；
- CFSR 的类投影空间、类自编码器和重构拒识；
- 新增的双视图门控融合、重构排序损失和类别条件误差校准。

## 数据划分

默认使用 `GlobecomPOWDER/`：

- 训练：Day 1，set 1–4，仅已知体制；
- 校准：Day 1，set 5，仅已知体制；
- 测试：Day 2，set 1–5，已知和未知体制；
- 同一录音切出的窗口不会跨集合。

每次运行选择一个未知体制，其余两个作为已知体制：

```powershell
python -m rat_cfsr.train --unknown 5G
python -m rat_cfsr.train --unknown 4G
python -m rat_cfsr.train --unknown WiFi
```

## 环境

建议 Python 3.10–3.12。安装与检查：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
python -m rat_cfsr.inspect_data --data-root GlobecomPOWDER
pytest
```

如需 GPU，请按本机 CUDA 版本安装 PyTorch，再执行 `pip install -e ".[test]"`。

## 快速验证

只构建数据和模型、执行一次前向传播：

```powershell
python -m rat_cfsr.train `
  --data-root GlobecomPOWDER `
  --unknown 5G `
  --max-windows-per-recording 4 `
  --batch-size 4 `
  --num-iq-samples 2048 `
  --n-fft 128 `
  --hop-length 64 `
  --dry-run
```

## 完整训练

```powershell
python -m rat_cfsr.train `
  --data-root GlobecomPOWDER `
  --output-dir outputs\unknown_5g_seed42 `
  --unknown 5G `
  --stage1-epochs 10 `
  --stage2-epochs 20 `
  --seed 42
```

输出目录包含：

- `checkpoint.pt`：模型权重和类别配置；
- `calibrator.json`：每类校准误差和阈值；
- `metrics.json`：AUROC、AUPR、OSCR、TKR、TUR 等；
- `history.json`：两阶段训练曲线；
- `test_predictions.npz`：测试标签、分数和重构误差；
- `split_summary.json`：录音及窗口划分统计。

## 重要说明

POWDER 中 Wi-Fi 原始采样率为 5 MS/s，LTE/5G NR 为 7.69 MS/s。数据集按照相同真实时间截窗，并把每个窗口重采样到相同输入长度，防止模型直接利用原始采样率或张量长度判断类别。

`S3R/` 保留官方源码用于方法追溯，不是新训练入口。新实现入口统一为 `rat_cfsr.train`。
