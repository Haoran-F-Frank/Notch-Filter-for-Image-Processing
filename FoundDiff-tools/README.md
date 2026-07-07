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

## HU 逆变换（结合体数据时很重要）

官方 `Normalize` 先做 `hu - 1024`，再线性映射。逆变换必须是：

- 正确：`hu = norm * 3000 + 24`
- 错误：`hu = norm * 3000 - 1000`（与原始 HU 相差 1024，叠成 3D 会在层边界错位）

**注意：** 把 `/path/to/your_input.nii.gz` 换成真实路径，那是文档占位符，不是真文件。
