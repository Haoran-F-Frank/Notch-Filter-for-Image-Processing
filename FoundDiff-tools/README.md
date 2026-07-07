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
python denoise_nii.py \
  --in_nii  /path/to/input.nii.gz \
  --out_nii data/Output/denoised.nii.gz \
  --z_start 100 --z_end 110
```
