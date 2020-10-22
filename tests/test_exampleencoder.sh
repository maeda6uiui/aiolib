#!/bin/bash                                                                                                                                                                                   

python3 ./test_exampleencoder.py \
    --train_example_filepath ~/AIOData/train_questions.json \
    --dev1_example_filepath ~/AIOData/dev1_questions.json \
    --dev2_example_filepath ~/AIOData/dev2_questions.json \
    --leaderboard_example_filepath ~/AIOData/aio_leaderboard.json \
    --context_filepath ~/AIOData/candidate_entities.json.gz \
    --bert_model_dir USE_DEFAULT \
    --train_save_dir ~/EncodedText/Train \
    --dev1_save_dir ~/EncodedText/Dev1 \
    --dev2_save_dir ~/EncodedText/Dev2 \
    --leaderboard_save_dir ~/EncodedText/Leaderboard
