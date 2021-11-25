#/bin/bash

export CUDA_VISIBLE_DEVICES=0

python run.py \
    --dataset=ml-1m \
    --train_dir=default \
    --inference_only=false \
    --maxlen=50 \
    --time_span=2048 \
    --eval_freq=20 \
#    --state_dict_path="./ml-1m_default/TiSASRec.epoch=5.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"