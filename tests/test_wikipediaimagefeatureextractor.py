import argparse
import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from util.image.frcnn import WikipediaImageFeatureExtractor

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

def main(
    article_list_filepath:str,
    image_root_dir:str,
    boxes_save_dir:str,
    features_save_dir:str):
    extractor=WikipediaImageFeatureExtractor(article_list_filepath)
    extractor.extract(image_root_dir,boxes_save_dir,features_save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--article_list_filepath",type=str)
    parser.add_argument("--image_root_dir",type=str)
    parser.add_argument("--boxes_save_dir",type=str)
    parser.add_argument("--features_save_dir",type=str)

    args=parser.parse_args()

    main(
        args.article_list_filepath,
        args.image_root_dir,
        args.boxes_save_dir,
        args.features_save_dir
    )
