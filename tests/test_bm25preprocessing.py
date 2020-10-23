import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from util.bm25.preprocessing import BM25Preprocessing

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(context_filepath:str,save_dir:str):
    preprocessing=BM25Preprocessing()
    preprocessing.preprocess(context_filepath,save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--context_filepath",type=str)
    parser.add_argument("--save_dir",type=str)

    args=parser.parse_args()

    main(args.context_filepath,args.save_dir)
