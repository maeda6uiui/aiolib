import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from util.image.frcnn import ImageFeatureExtractor

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(
    image_root_dir:str,
    boxes_save_dir:str,
    features_save_dir:str,
    index_lower_bound:int,
    index_upper_bound:int):
    extractor=ImageFeatureExtractor()
    extractor.extract(
        image_root_dir,
        boxes_save_dir,
        features_save_dir,
        index_lower_bound=index_lower_bound,
        index_upper_bound=index_upper_bound
    )

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_root_dir",type=str)
    parser.add_argument("--boxes_save_dir",type=str)
    parser.add_argument("--features_save_dir",type=str)
    parser.add_argument("--index_lower_bound",type=int,default=-1)
    parser.add_argument("--index_upper_bound",type=int,default=-1)
    args=parser.parse_args()

    main(
        args.image_root_dir,
        args.boxes_save_dir,
        args.features_save_dir,
        args.index_lower_bound,
        args.index_upper_bound
    )
