import argparse
import logging
import sys
import os
import torch
sys.path.append(os.path.abspath("../src/aiolib"))

from modeling.baseline.modeling import BaselineModeler
from util.seed import set_seed

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(
    train_input_dir:str,
    dev_input_dir:str,
    bert_model_dir:str,
    result_save_dir:str):
    set_seed(42)

    modeler=BaselineModeler(train_input_dir,dev_input_dir,bert_model_dir)
    modeler.to(device)
    modeler.train_and_eval(result_save_dir=result_save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--train_input_dir",type=str)
    parser.add_argument("--dev_input_dir",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--result_save_dir",type=str)

    args=parser.parse_args()

    main(
        args.train_input_dir,
        args.dev_input_dir,
        args.bert_model_dir,
        args.result_save_dir
    )
