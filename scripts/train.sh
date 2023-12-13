#!/bin/bash

# For GPU machines

# python -m open_ml.training.main \
torchrun --nproc-per-node 2 -m open_ml.training.main \
        --train-data "./data/mnist-train{0..7}.tar" \
        --train-num-samples 2000 \
        --val-data "./data/mnist-val.tar" \
        --val-num-samples 100 \
	--dataset-type "webdataset" \
        --batch-size 2 \
        --workers 1 \
        --accum-freq 2 \
        --log-every-n-steps 10 \
        --grad-clip-norm 1 \
        --lr 3e-3 \
        --lr-cooldown-end 3e-4 \
        --wd 0.1 \
        --model "mlp_mini" \
	--resume "logs/test_resume/checkpoints/epoch_3.pt" \
	--fsdp \
        --precision amp_bfloat16 \
        --beta2 0.95 \
        --warmup 50 \
        --epochs 10 \
        --delete-previous-checkpoint \
