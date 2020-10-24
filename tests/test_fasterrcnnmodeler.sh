#!/bin/bash

python3 ./test_fasterrcnnmodeler.py \
    --train_input_dir ~/EncodedText/Train \
    --dev_input_dir ~/EncodedText/Dev1 \
    --bert_model_dir USE_DEFAULT \
    --im_features_dir ~/FasterRCNNFeatures2/Projected \
    --result_save_dir ./OutputDir/FasterRCNN
