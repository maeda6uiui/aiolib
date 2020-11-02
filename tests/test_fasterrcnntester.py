import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from modeling.frcnn.modeling import FasterRCNNTester

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(
    test_input_dir:str,
    bert_model_dir:str,
    im_boxes_dir:str,
    im_features_dir:str,
    checkpoint_dir:str,
    result_save_dir:str):
    tester=FasterRCNNTester(
        test_input_dir,
        bert_model_dir,
        im_boxes_dir,
        im_features_dir
    )

    for i in range(5):
        checkpoint_filepath=os.path.join(checkpoint_dir,"checkpoint_{}.pt".format(i+1))
        result_filepath=os.path.join(result_save_dir,"result_test_{}.txt".format(i+1))
        labels_filepath=os.path.join(result_save_dir,"labels_test_{}.txt".format(i+1))

        tester.test(checkpoint_filepath,result_filepath,labels_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--test_input_dir",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--im_boxes_dir",type=str)
    parser.add_argument("--im_features_dir",type=str)
    parser.add_argument("--checkpoint_dir",type=str)
    parser.add_argument("--result_save_dir",type=str)

    args=parser.parse_args()

    main(
        args.test_input_dir,
        args.bert_model_dir,
        args.im_boxes_dir,
        args.im_features_dir,
        args.checkpoint_dir,
        args.result_save_dir
    )
