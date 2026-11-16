# 本地部署补丁记录（Apple Silicon / MPS）

2026-11-16 在 M5 Max（macOS, 无 CUDA）上完成 AnySD 推理部署所打的补丁。
上游代码是 CUDA 向的，以下改动使其可在 MPS 上运行。demo 输出见 `assets/anysd-test/`。

## 代码改动

1. `anysd/src/model.py`
   - 顶部新增 `DEVICE` / `DTYPE` 常量（环境变量 `ANYSD_DEVICE` / `ANYSD_DTYPE` 可覆盖，
     默认 mps + float16）。
   - 4 处 `.to("cuda")` / `torch.Generator("cuda")` 改为 `DEVICE`（两条流水线 + generator）。
   - `_process_ref_image_cfg` 中 2 处 `encode_image(..., "cuda", ...)` 改为 `DEVICE`，
     `torch.zeros` 增加 `device=DEVICE`（visual 系任务用）。

2. `anysd/src/utils.py`
   - `get_experts_dir` 由「只下 README 再拼路径」改为 `snapshot_download(allow_patterns=["experts/*"])`，
     否则 `load_ip_adapter` 找不到 experts/*.bin。
   - 新增 `import os`。

3. `anysd/infer.py`
   - 去掉 `from anysd.train.valid_log import download_image`（该模块拖入 wandb/clip/anybench
     评测依赖），本地用 PIL 实现 `download_image`。
   - 推理步数与用例数可用环境变量控制：`ANYSD_STEPS`（默认 100）、`ANYSD_MAX_CASES`（默认全部 7 例）。

## 运行方式

```bash
cd ~/Developer/AnySD
PYTHONPATH=./ HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
  .venv/bin/python -u anysd/infer.py
# 输出到 assets/anysd-test/；MPS fp16 约 4 it/s，100 步/张约 25s
```

## 权重（HF 缓存，共约 10GB）

- `stable-diffusion-v1-5/stable-diffusion-v1-5`（fp16 权重 ~1.9GB）
- `WeiChow/AnySD`：unet fp32 3.4GB、image_encoder 3.69GB、experts/*.bin ×10（各 197MB）

## 已踩过的坑

- **hf_xet 挂死**：`image_encoder/model.safetensors`（3.69GB）经 HF 库下载两次都在 ~1.5GB 处
  卡死（hf_xet 与 hf_transfer 皆然）。最终用 curl 8 段并行下载 + sha256 校验 + 手工放入
  HF 缓存（blob 名即文件 sha256，snapshot 下建相对符号链接）解决。
  `HF_HUB_DISABLE_XET=1` 可避免 xet 后端；下载大文件推荐直连 + 分段。
- 代理：macOS 系统代理（127.0.0.1:7890）开着时 Python 会走代理；本网络直连
  huggingface.co 也可用，两者速度相近（~1–3MB/s）。

## 已知边界

- 仅推理可用；训练 / 数据构建流水线（`scripts/*.sh`）仍需 CUDA（xformers、bitsandbytes）。
- fp16 在 MPS 偶发黑图时，用 `ANYSD_DTYPE=float32` 重跑（显存足够，速度约慢一倍）。
- 上游 `AnySDPipeline._shift_adapter` 里 `'gerneral'` 为上游拼写错误，general 分支走
  `general_sd_pipe` 不受影响。
