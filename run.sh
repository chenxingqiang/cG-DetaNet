#!/bin/bash

# 设置参数
dataset_name="qm9"
datapath="data/qm9/"
modelpath="models/detanet_qm9/"
split_path="data/qm9/split.npz"
seed=42
features=128
num_radial=32
cutoff=5.0
num_block=3
maxl=3
lr=1e-4
batch_size=128
max_epochs=1000

# 训练模型
python detanet_cond_script.py train \
--model detanet \
--dataset_name "${dataset_name}" \
--datapath "${datapath}" \
--modelpath "${modelpath}" \
--split_path "${split_path}" \
--seed "${seed}" \
--features "${features}" \
--num_radial "${num_radial}" \
--cutoff "${cutoff}" \
--num_block "${num_block}" \
--maxl "${maxl}" \
--lr "${lr}" \
--batch_size "${batch_size}" \
--max_epochs "${max_epochs}"

# 在测试集上评估模型
python detanet_cond_script.py eval \
--model detanet \
--datapath "${datapath}" \
--modelpath "${modelpath}" \
--split test

# 使用训练好的模型生成分子
amount=1000
temp=0.1
python detanet_cond_script.py generate \
--model detanet \
--modelpath "${modelpath}" \
--amount "${amount}" \
--temperature "${temp}"