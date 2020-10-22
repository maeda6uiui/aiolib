#!/bin/bash

python3 ./test_projector.py \
    --src_dim 1024 \
    --dst_dim 768 \
    --src_dir ~/FasterRCNNFeatures/Original \
    --save_dir ~/FasterRCNNFeatures/Projected
