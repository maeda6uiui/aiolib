import argparse
import cv2
import glob
import logging
import numpy as np
import os
import torch
from typing import Tuple

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_region_features_single(
    raw_image:np.ndarray,
    predictor:DefaultPredictor)->Tuple[torch.Tensor,torch.Tensor]:
    """
    一つの画像から特徴量を取得する。
    """
    with torch.no_grad():
        raw_height,raw_width=raw_image.shape[:2]

        image=predictor.aug.get_transform(raw_image).apply_image(raw_image)
        image=torch.as_tensor(image.astype("float32").transpose(2,0,1))
        inputs=[{"image":image,"height":raw_height,"width":raw_width}]
        images=predictor.model.preprocess_image(inputs)

        model=predictor.model

        #Backboneから特徴量を生成する。
        feature=model.backbone(images.tensor)
        #Proposalを生成する。
        proposals,_=model.proposal_generator(images,feature)
        instances,_=model.roi_heads(images,feature,proposals)
        #RoI特徴量を生成する。
        pred_boxes=[x.pred_boxes for x in instances]
        box_features=model.roi_heads.box_pooler(
            [feature[f] for f in feature if f!="p6"],pred_boxes
        )
        box_features=model.roi_heads.box_head(box_features) #FC層の出力 (RoIの数,特徴量の次元数)

        box_coords=torch.empty(0,4).to(device)    #RoIの座標
        for boxes in pred_boxes:
            box_coords=torch.cat([box_coords,boxes.tensor],dim=0)

        return box_coords,box_features

def main(args):
    image_dir:str=args.image_dir
    boxes_save_dir:str=args.boxes_save_dir
    features_save_dir:str=args.features_save_dir
    index_lower_bound:int=args.index_lower_bound
    index_upper_bound:int=args.index_upper_bound

    model_name="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
    cfg.MODEL.DEVICE=str(device)
    cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url(model_name)
    predictor=DefaultPredictor(cfg)

    os.makedirs(boxes_save_dir,exist_ok=True)
    os.makedirs(features_save_dir,exist_ok=True)

    pathname=os.path.join(image_dir,"*.jpg")
    files=glob.glob(pathname)
    for idx,file in enumerate(files):
        if idx<index_lower_bound:
            continue
        if index_upper_bound>=0 and idx>=index_upper_bound:
            break

        logger.info("{}\t{}".format(idx,file))

        image=cv2.imread(file)
        if image is None:
            logger.warn("画像を開けませんでした。\t{}".format(file))
        
        boxes,features=get_region_features_single(image,predictor)

        save_filename=os.path.basename(os.path.splitext(file)[0])+".pt"
        boxes_save_filepath=os.path.join(boxes_save_dir,save_filename)
        features_save_filepath=os.path.join(features_save_dir,save_filename)

        torch.save(boxes,boxes_save_filepath)
        torch.save(features,features_save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_dir",type=str)
    parser.add_argument("--boxes_save_dir",type=str)
    parser.add_argument("--features_save_dir",type=str)
    parser.add_argument("--index_lower_bound",type=int,default=-1)
    parser.add_argument("--index_upper_bound",type=int,default=-1)
    args=parser.parse_args()

    main(args)
