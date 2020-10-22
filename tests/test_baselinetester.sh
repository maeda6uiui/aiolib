#!/bin/bash

python3 ./test_baselinetester.py \
    --test_input_dir ~/EncodedText/Dev2 \
    --bert_model_dir USE_DEFAULT \
    --checkpoint_dir ./OutputDir/Baseline \
    --result_save_dir ./OutputDir/Baseline
