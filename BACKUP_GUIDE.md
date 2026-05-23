# SOAR 信息备份指南

本指南用于备份 SOAR 项目中除模型权重和运行环境以外的可复现信息。目标是保留研究过程、脚本、提交包、结果摘要、日志和本地 patch；不备份可重新下载或重新构建的大体积资产。

## 默认备份范围

推荐备份：

- 根仓库文件：`README.md`、`AGENTS.md`、`.gitignore`、`BACKUP_GUIDE.md`。
- Git 历史和当前 working tree：包含 `.git/`、已修改文件、未跟踪结果文件。
- 本地脚本：`scripts/`。
- 提交包骨架：`submit/`。
- 实验摘要和轻量结果：`artifacts/results/`。
- 运行日志：`artifacts/logs/`。日志不进 git，但对复盘环境安装和 benchmark 失败有用，当前体积很小。
- 第三方源码的本地差异 patch：`submit/patches/`。

默认不备份：

- 模型权重：`/home/dataset-local/models/`、`*.safetensors`、`*.bin`、`*.pt`、`*.pth`、`*.ckpt`、`*.gguf`、`*.onnx`。
- Python 环境和包缓存：`.venv/`、`/home/dataset-local/.cache/`、`uv` cache、Hugging Face cache。
- CUDA toolkit 或系统环境：例如 `/home/dataset-local/cuda-13.1`。
- 大型可再生产物：`artifacts/checkpoints/`、`artifacts/draft_heads/`。
- 第三方源码 checkout 本体：`repos/` 默认不打包，只记录 remote、branch、commit 和必要 patch。

## 当前第三方源码定位信息

如果以后需要重建 `repos/`，优先按下面信息重新 clone/checkout，再应用 `submit/patches/` 中的 patch。

| 路径 | Remote | Branch | Commit |
|---|---|---|---|
| `repos/SOAR-Toolkit` | `https://github.com/OpenBMB/SOAR-Toolkit.git` | `soar-rebuild-eagle3-lk` | `2ed4ade` |
| `repos/SpecForge` | `https://github.com/sgl-project/SpecForge.git` | `soar-rebuild-eagle3-lk` | `d5fb617` |
| `repos/sglang` | `https://github.com/sgl-project/sglang.git` | `soar-rebuild-eagle3-lk` | `791a2f0` |
| `repos/openbmb_sglang_minicpm_sala` | `https://github.com/OpenBMB/sglang.git` | `minicpm_sala` | `d29fb13` |

`repos/openbmb_sglang_minicpm_sala` 的 submodule 状态：

| Submodule | Commit | 本地差异 |
|---|---:|---|
| `3rdparty/infllmv2_cuda_impl` | `ed98dfc` | `submit/patches/infllmv2_cuda_impl_setup_cuda_arch_override.patch` |
| `3rdparty/infllmv2_cuda_impl/csrc/cutlass` | `4c42f73` | `submit/patches/cutlass_cuda_13_host_adapter.patch` |
| `3rdparty/sparse_kernel` | `d7c367e` | `submit/patches/sparse_kernel_setup_cuda_arch_override.patch` |

## 生成轻量备份包

这条命令会在 `/home/dataset-local/backups/SOAR/` 生成一个信息备份包。注意：这只是本机 staging，真正备份还要复制到另一块盘、另一台机器或云端。

```bash
mkdir -p /home/dataset-local/backups/SOAR
backup=/home/dataset-local/backups/SOAR/SOAR-info-backup-$(date +%Y%m%d_%H%M%S).tar.gz

tar -C /home/dataset-local/work \
  --exclude='SOAR/.venv' \
  --exclude='SOAR/repos' \
  --exclude='SOAR/artifacts/checkpoints' \
  --exclude='SOAR/artifacts/draft_heads' \
  --exclude='SOAR/outputs' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.gguf' \
  --exclude='*.onnx' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  -czf "$backup" SOAR

sha256sum "$backup" > "$backup.sha256"
tar -tzf "$backup" > "$backup.manifest.txt"
```

## 快速验包

```bash
tar -tzf "$backup" | grep -E '(^SOAR/\.venv/|^SOAR/repos/|^SOAR/artifacts/checkpoints/|^SOAR/artifacts/draft_heads/|\.(safetensors|bin|pt|pth|ckpt|gguf|onnx|tar|tar\.gz|zip)$)' \
  && echo "unexpected large/model/env file found" \
  || echo "backup scope looks clean"

sha256sum -c "$backup.sha256"
```

## 复制到真正的备份位置

同机备份不能防磁盘故障。生成包以后，至少复制到一个外部位置：

```bash
rsync -avh --progress "$backup" "$backup.sha256" "$backup.manifest.txt" user@backup-host:/path/to/backups/SOAR/
```

如果是移动硬盘，把目标路径换成挂载目录即可：

```bash
rsync -avh --progress "$backup" "$backup.sha256" "$backup.manifest.txt" /mnt/backup/SOAR/
```

## 可选：离线源码备份

如果预计以后没有网络，才额外备份 `repos/`。这会多约 456 MB，不含模型和 `.venv`。

```bash
offline_backup=/home/dataset-local/backups/SOAR/SOAR-source-offline-$(date +%Y%m%d_%H%M%S).tar.gz

tar -C /home/dataset-local/work \
  --exclude='SOAR/.venv' \
  --exclude='SOAR/artifacts/checkpoints' \
  --exclude='SOAR/artifacts/draft_heads' \
  --exclude='SOAR/outputs' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.gguf' \
  --exclude='*.onnx' \
  -czf "$offline_backup" SOAR

sha256sum "$offline_backup" > "$offline_backup.sha256"
tar -tzf "$offline_backup" > "$offline_backup.manifest.txt"
```

## 恢复检查

```bash
mkdir -p /tmp/soar-restore-test
tar -xzf "$backup" -C /tmp/soar-restore-test
cd /tmp/soar-restore-test/SOAR
git status --short --branch
```

恢复后如果需要重新构建运行环境，再按 `README.md` 和 `AGENTS.md` 里的环境步骤创建 `.venv`、下载模型、clone `repos/`。不要从备份包里恢复旧 `.venv` 或旧模型权重。
