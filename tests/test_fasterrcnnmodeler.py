import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from modeling.baseline.modeling import FasterRCNNModeler

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(
    train_input_dir:str,
    dev_input_dir:str,
    bert_model_dir:str,
    im_features_dir:str,
    result_save_dir:str):
    modeler=FasterRCNNModeler(train_input_dir,dev_input_dir,bert_model_dir,im_features_dir)
    modeler.train_and_eval(result_save_dir=result_save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--train_input_dir",type=str)
    parser.add_argument("--dev_input_dir",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--im_features_dir",type=str)
    parser.add_argument("--result_save_dir",type=str)

    args=parser.parse_args()

    main(
        args.train_input_dir,
        args.dev_input_dir,
        args.bert_model_dir,
        args.im_features_dir,
        args.result_save_dir
    )