#!/bin/bash

model=llava-v1.5-13b
expansion_rate=1.0
attn=4.5
perturb_weight=1


exp_name=$model-CoF-mme-$attn-$expansion_rate

mkdir -p /sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/MME/answers/$exp_name
touch /sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/MME/answers/$exp_name/$exp_name.jsonl
touch /sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/MME/answers/$exp_name/eval.log

python examples/eval_scripts/llava_vqa_loader_hl.py \
    --model-path /sda/gaodh/projects/CoF/checkpoints/$model \
    --question-file base_models/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder base_models/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file base_models/LLaVA/playground/data/eval/MME/answers/$exp_name/$exp_name.jsonl \
    --temperature 0 \
    --attn $attn \
    --perturb_weight $perturb_weight \
    --expansion_rate $expansion_rate \
    --conv-mode vicuna_v1

cd /sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/MME

python convert_answer_to_mme.py --experiment $exp_name

cd /sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/MME/eval_tool

python calculation.py --results_dir /sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/MME/answers/$exp_name > /sda/gaodh/projects/CoF/base_models/LLaVA/playground/data/eval/MME/answers/$exp_name/eval.log