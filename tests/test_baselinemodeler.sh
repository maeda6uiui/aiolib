#!/bin/bash

python3 ./test_baselinemodeler.py \
    --train_input_dir ~/EncodedText/Train \
    --dev_input_dir ~/EncodedText/Dev1 \
    --bert_model_dir USE_DEFAULT \
    --result_save_dir ./OutputDir/Baseline
