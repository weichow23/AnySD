# AnyBench and AnySD

[![arXiv](https://img.shields.io/badge/arXiv-2411.15738-b31b1b.svg)](https://arxiv.org/abs/2411.15738)
[![Dataset](https://img.shields.io/badge/🤗%20Huggingface-Dataset-yellow)](https://huggingface.co/datasets/Bin1117/AnyEdit)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/DCDmllm/AnyEdit)
[![Page](https://img.shields.io/badge/Home-Page-b3.svg)](https://dcd-anyedit.github.io/)


> This is the official model implementation and benchmark evaluation repository of 
> **AnyEdit: Unified High-Quality Image Edit with Any Idea**

## 📊 AnyBench

This is the guide for the evaluation tool for AnyBench. The specific files are located in the `anybench` directory.

We have integrated the evaluations for `AnyBench`, `Emu-edit`, and `MagicBrush` into the same codebase, and it supports the following models: `Null-Text`, `Uni-ControlNet`, `InstructPix2Pix`, `MagicBrush`, `HIVE`, and `UltraEdit (SD3)`.

Evaluation metrics are `CLIPim↑`, ` CLIPout↑`, ` L1↓` ,` L2↓`and  `DINO↑`

#### **🚀** Quick Start

```shell
bash anybench/setup.sh  # You need to go into the script and carefully check to ensure that the correct dependencies are installed.
```

#### 🏆 Evaluation

**EMU-Edit**

1. download dataset via

```shell
huggingface-cli download facebook/emu_edit_test_set_generations --repo-type dataset
```

2. run

```shell
# gen images
CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' python anybench/eval/emu_gen.py
# test
CUDA_VISIBLE_DEVICES=3 PYTHONPATH='./' python anybench/eval/emu_eval.py
```

**MagicBrush**

download the test set from [MagicBrush](https://osu-nlp-group.github.io/MagicBrush/)

```shell
CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' python anybench/eval/magicbrush_gen_eval.py
```

**AnyBench**

```shell
CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' python anybench/eval/anybench_gen_eval.py
```

⚠ Notice: AnySD may output completely black images for certain sensitive commands, which is a normal occurrence.

⚠ Notice: During evaluation, the final scores may vary due to the influence of inference hyperparameters, random seeds, and batch size.

## 🎨 AnySD

#### **🚀** Quick Start

1. **Clone this repo**

```shell
vim ~/.bashrc
export HF_HOME=/mnt/bn/magellan-product-audit/weic/data_hf
#按 Esc 退出插入模式, 输入 :wq 保存并退出 vim
source ~/.bashrc
echo $HF_HOME

git clone https://github.com/weichow23/AnyDM

git add .
git commit -m "update"
git push origin main
```


2. **Environment setup**
```bash
conda create -n anyedit python=3.9
conda activate anyedit
pip install -r requirements.txt
pip install --upgrade torch diffusers xformers triton pydantic deepspeed
pip install git+https://github.com/openai/CLIP
```


#### 🌐 Inference

```shell

```

#### 🔮 Training

1. **Stage I**
```shell
bash train_stage1.sh
```
2. **Stage II**
```

```


#### 🔎 Summary

Since  **AnyEdit** contains a wide range of editing instructions across various domains, it holds promising potential for developing a powerful editing model to address high-quality editing tasks. However, training such a model has three extra challenges: (a) aligning the semantics of various multi-modal inputs; (b) identifying the semantic edits within each domain to control the granularity and scope of the edits; (c) coordinating the complexity of various editing tasks to prevent catastrophic forgetting. To this end, we propose a novel **AnyEdit Stable Diffusion** approach (🎨**AnySD**) to cope with various editing tasks in the real world.

<img src='assets/model.png' width='100%' />

**Architecture of 🎨AnySD**. 🎨**AnySD** is a novel architecture that supports three conditions (original image, editing instruction, visual prompt) for various editing tasks.

💖 Our model is based on the awesome **[SD 1.5 ](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)**

## 📚 Citation

```shell
@article{yu2024anyedit,
  title={AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea},
  author={Yu, Qifan and Chow, Wei and Yue, Zhongqi and Pan, Kaihang and Wu, Yang and Wan, Xiaoyang and Li, Juncheng and Tang, Siliang and Zhang, Hanwang and Zhuang, Yueting},
  journal={arXiv preprint arXiv:2411.15738},
  year={2024}
}
```
