# CoF: Coarse to Fine-Grained Image Understanding for Multi-modal Large Language Models

[CoF: Coarse to Fine-Grained Image Understanding for Multi-modal Large Language Models](https://arxiv.org/pdf/2412.16869) (published at ICASSP 2025).

## Introduction

This is the official implementation of the paper *CoF: Coarse to Fine-Grained Image Understanding for Multi-modal Large Language Models*.  In this paper,  we propose 
 Coarse to Fine (CoF) to break down multi-modal understanding into two steps. In the first stage, we prompt the MLLM to locate the approximate area of the answer. In the second stage, we further enhance the model’s focus on relevant areas within the image through visual prompt engineering, adjusting attention weights of pertinent regions.

## Quick Start

Basic enviornment setup:

```zsh
conda create -n cof python=3.10 -y
conda activate cof
pip install -r requirements.txt
```
### LLaVA

Install latest LLaVA model `2023-11-30` in base_models. If you already have one, you can use the installed one in your own enviornment.

```bash
# you may also use your installed llava if you have installed.
cd base_models
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

**Model Download**: Please refer to [LLaVAv1.5 Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) to get the base pretrained model.

For MME:

```bash
bash examples/eval_scripts/mme_hl.sh
```

## Citation
Please cite the following paper if you find the repo helpful:
```
@misc{wang2024cofcoarsefinegrainedimage,
      title={CoF: Coarse to Fine-Grained Image Understanding for Multi-modal Large Language Models}, 
      author={Yeyuan Wang and Dehong Gao and Bin Li and Rujiao Long and Lei Yi and Xiaoyan Cai and Libin Yang and Jinxia Zhang and Shanqing Yu and Qi Xuan},
      year={2024},
      eprint={2412.16869},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.16869}, 
}
```

## Acknowledgement

We would like to thank the following repos for their great work:

- This work utilizes multi-modal LLMs with base models in [LLaVA](https://github.com/haotian-liu/LLaVA) and [InstructBLIP](https://github.com/salesforce/LAVIS).
- This work utilizes the attention highlighter referenced in [Prompt Highlighter](https://github.com/dvlab-research/Prompt-Highlighter).
