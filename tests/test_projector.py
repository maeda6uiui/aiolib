import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from util.image.projector import Projector

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(
    src_dim:int,
    dst_dim:int,
    src_dir:str,
    save_dir:str):
    projector=Projector(src_dim,dst_dim)
    projector.project_from_directory(src_dir,save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--src_dim",type=int)
    parser.add_argument("--dst_dim",type=int)
    parser.add_argument("--src_dir",type=str)
    parser.add_argument("--save_dir",type=str)

    args=parser.parse_args()

    main(
        args.src_dim,
        args.dst_dim,
        args.src_dir,
        args.save_dir
    )
