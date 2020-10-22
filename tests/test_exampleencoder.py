import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from util.encode import ExampleEncoder

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(
    train_example_filepath:str,
    dev1_example_filepath:str,
    dev2_example_filepath:str,
    leaderboard_example_filepath:str,
    context_filepath:str,
    bert_model_dir:str,
    train_save_dir:str,
    dev1_save_dir:str,
    dev2_save_dir:str,
    leaderboard_save_dir:str):
    encoder=ExampleEncoder(context_filepath,bert_model_dir)
    encoder.encode_save(train_example_filepath,train_save_dir)
    encoder.encode_save(dev1_example_filepath,dev1_save_dir)
    encoder.encode_save(dev2_example_filepath,dev2_save_dir)
    encoder.encode_save(leaderboard_example_filepath,leaderboard_save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--train_example_filepath",type=str)
    parser.add_argument("--dev1_example_filepath",type=str)
    parser.add_argument("--dev2_example_filepath",type=str)
    parser.add_argument("--leaderboard_example_filepath",type=str)
    parser.add_argument("--context_filepath",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--train_save_dir",type=str)
    parser.add_argument("--dev1_save_dir",type=str)
    parser.add_argument("--dev2_save_dir",type=str)
    parser.add_argument("--leaderboard_save_dir",type=str)

    args=parser.parse_args()

    main(
        args.train_example_filepath,
        args.dev1_example_filepath,
        args.dev2_example_filepath,
        args.leaderboard_example_filepath,
        args.context_filepath,
        args.bert_model_dir,
        args.train_save_dir,
        args.dev1_save_dir,
        args.dev2_save_dir,
        args.leaderboard_save_dir
    )
