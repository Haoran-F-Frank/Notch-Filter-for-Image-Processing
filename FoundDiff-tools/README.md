# FoundDiff 本地同步工具

从 Cloud Agent 同步到本机 `~/FoundDiff2.0/FoundDiff`，**无需 sudo、无需 wget 认证**。

## 一次性设置

```bash
cd ~/FoundDiff2.0
# 若还没有官方仓库，先克隆
git clone https://github.com/hao1635/FoundDiff.git FoundDiff
cd FoundDiff

# 添加上游工具仓库（公开，git clone 无需登录）
git remote add tools https://github.com/Haoran-F-Frank/Notch-Filter-for-Image-Processing.git
git fetch tools founddiff-tools-7d4e
```

## 每次更新（推荐）

在 `~/FoundDiff2.0/FoundDiff` 目录执行：

```bash
cd ~/FoundDiff2.0/FoundDiff
git fetch tools founddiff-tools-7d4e
git checkout tools/founddiff-tools-7d4e -- FoundDiff-tools/
bash FoundDiff-tools/sync.sh
```

`sync.sh` 会把脚本复制到仓库根目录，并对 `src/DADiff.py` 打补丁。

## 仅克隆工具文件（不用 git remote）

```bash
cd ~/FoundDiff2.0/FoundDiff
git clone --depth 1 --branch founddiff-tools-7d4e \
  https://github.com/Haoran-F-Frank/Notch-Filter-for-Image-Processing.git /tmp/notch-tools
cp /tmp/notch-tools/FoundDiff-tools/*.py .
cp /tmp/notch-tools/FoundDiff-tools/sync.sh .
bash sync.sh
rm -rf /tmp/notch-tools
```

## 包含的改动

| 文件 | 说明 |
|------|------|
| `demo_denoise.py` | 单层 .npy 去噪；`init()` 后 `ema.to(device)` 修复 CUDA 错误 |
| `denoise_nii.py` | 端到端 nii.gz → 去噪 → nii.gz + 每层 stats CSV |
| `convert_nii_to_npy.py` | nii 转 npy 辅助 |
| `src/DADiff.py` | 支持 `DA-CLIP.pth` 回退路径 |

## 运行示例

```bash
# 10 层测试（约 2-4 分钟，batch_size=1）
python denoise_nii.py \
  --in_nii  data/mydata/APNHC00002_CT.nii.gz \
  --out_nii data/Output/APNHC00002_CT_test.nii.gz \
  --z_start 100 --z_end 110

# 加速：显存够用时增大 batch（RTX 可试 2 或 4）
python denoise_nii.py \
  --in_nii  data/mydata/APNHC00002_CT.nii.gz \
  --out_nii data/Output/APNHC00002_CT.nii.gz \
  --batch_size 2 --skip_air

# 速度参考：~25 s/层 (batch=1)，480 层约 3 小时
```

## HU 强度映射（已改为你的公式）

**不再使用**作者原版 `(hu - 1024 + 1000) / 3000`。

当前 `intensity.py`：

```
norm = clip((hu + 2000) / 3000,  0, 1)   # HU [-2000, +1000] -> [0, 1]
hu   = norm * 3000 - 2000
```

| HU | norm |
|----|------|
| -2000 | 0 |
| -1000（空气） | 0.33 |
| 0 | 0.67 |
| +1000 | 1 |

> 注意：与作者训练时的 Normalize 不同，可能影响去噪效果。若结果异常可再调 `HU_ADD`/`HU_RANGE`。

## 强度范围线性匹配（默认开启）

每层记录原始 min/max/range，去噪后再线性映射回原始强度域：

```
matched = scale * denoised_raw + offset
scale   = (in_max - in_min) / (out_raw_max - out_raw_min)
offset  = in_min - scale * out_raw_min
```

| 参数 | 说明 |
|------|------|
| `--match_intensity slice` | 每层对齐到该层原始 min/max（**默认**） |
| `--match_intensity global` | 所有层对齐到整体原始 min/max |
| `--match_intensity none` | 不做匹配，保留模型输出 HU |
| `--out_nii_raw` | 另存匹配前的去噪结果 |

CSV 列：`in_min/max/range`, `out_raw_min/max/range`, `out_matched_min/max/range`, `match_scale`, `match_offset`, `range_ratio_raw_vs_in`

**注意：** 把 `/path/to/your_input.nii.gz` 换成真实路径，那是文档占位符，不是真文件。
