#!/bin/bash                                                                                                                                                                                   

python3 ./test_wikipediaimagefeatureextractor.py \
    --article_list_filepath ~/WikipediaImages/article_list.txt \
    --image_root_dir ~/WikipediaImages/Images \
    --save_dir ~/FasterRCNNFeatures
