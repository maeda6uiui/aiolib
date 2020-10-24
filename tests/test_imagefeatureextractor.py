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
    save_dir:str):
    extractor=ImageFeatureExtractor()
    extractor.extract(image_root_dir,save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_root_dir",type=str)
    parser.add_argument("--save_dir",type=str)
    args=parser.parse_args()

    main(args.image_root_dir,args.save_dir)
