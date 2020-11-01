import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from util.image.projector import Projector

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(
    features_src_dim:int,
    dst_dim:int,
    boxes_src_dir:str,
    boxes_save_dir:str,
    features_src_dir:str,
    features_save_dir:str):
    projector=Projector(features_src_dim,dst_dim)
    projector.project_from_directory(
        boxes_src_dir,
        boxes_save_dir,
        features_src_dir,
        features_save_dir
    )

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--features_src_dim",type=int)
    parser.add_argument("--dst_dim",type=int)
    parser.add_argument("--boxes_src_dir",type=str)
    parser.add_argument("--boxes_save_dir",type=str)
    parser.add_argument("--features_src_dir",type=str)
    parser.add_argument("--features_save_dir",type=str)

    args=parser.parse_args()

    main(
        args.features_src_dim,
        args.dst_dim,
        args.boxes_src_dir,
        args.boxes_save_dir,
        args.features_src_dir,
        args.features_save_dir
    )
