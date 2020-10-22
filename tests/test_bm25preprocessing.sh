#!/bin/bash

python3 ./test_bm25preprocessing.py \
    --context_filepath ~/AIOData/candidate_entities.json.gz \
    --save_dir ~/BM25Stats
