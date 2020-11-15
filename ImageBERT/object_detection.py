"""
Faster R-CNNを用いて物体検出を行う。
"""
import argparse
import cv2
import glob
import logging
import numpy as np
import os
import torch
from typing import List,Tuple

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
        features=model.backbone(images.tensor)
        #Proposalを生成する。
        proposals,_=model.proposal_generator(images,features)
        instances,_=model.roi_heads(images,features,proposals)
        #RoI特徴量を生成する。
        pred_boxes=[x.pred_boxes for x in instances]
        box_features=model.roi_heads.box_pooler(
            [features[f] for f in features if f!="p6"],pred_boxes
        )
        box_features=model.roi_heads.box_head(box_features) #FC層の出力 (RoIの数,特徴量の次元数)

        boxes_tensor=torch.empty(0,4).to(device)    #RoIの座標
        for boxes in pred_boxes:
            boxes_tensor=torch.cat([boxes_tensor,boxes.tensor],dim=0)

        return boxes_tensor,box_features

def get_region_features(
    raw_images:List[np.ndarray],
    predictor:DefaultPredictor)->Tuple[torch.Tensor,torch.Tensor]:
    """
    複数の画像から特徴量を取得する。
    物体が何も検出されなかった場合にはNoneが返される。
    """
    boxes_list=[]
    features_list=[]
    for raw_image in raw_images:
        boxes,features=get_region_features_single(raw_image,predictor)
        boxes_list.append(boxes)
        features_list.append(features)

    #Tensorを結合する。
    if len(features_list)==0:
        return None,None

    #RoIの座標
    ret_boxes=torch.empty(0,4).to(device)
    for boxes in boxes_list:
        ret_boxes=torch.cat([ret_boxes,boxes],dim=0)
    
    #RoIの特徴量
    dimension=features_list[0].size(1)
    ret_features=torch.empty(0,dimension).to(device)
    for features in features_list:
        ret_features=torch.cat([ret_features,features],dim=0)
    
    return ret_boxes,ret_features

def main(args):
    model_name:str=args.model_name
    image_root_dir:str=args.image_root_dir
    boxes_save_dir:str=args.boxes_save_dir
    features_save_dir:str=args.features_save_dir
    index_lower_bound:int=args.index_lower_bound
    index_upper_bound:int=args.index_upper_bound

    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
    cfg.MODEL.DEVICE=str(device)
    cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url(model_name)
    predictor=DefaultPredictor(cfg)

    #結果を保存するディレクトリを作成する。
    os.makedirs(boxes_save_dir,exist_ok=True)
    os.makedirs(features_save_dir,exist_ok=True)

    #処理を開始する。
    pathname=os.path.join(image_root_dir,"*")
    directories=glob.glob(pathname)
    for idx,directory in enumerate(directories):
        if idx<index_lower_bound:
            continue
        if index_upper_bound>=0 and idx>=index_upper_bound:
            break

        logger.info("{}\t{}".format(idx,directory))

        pathname=os.path.join(directory,"*[!txt]")
        files=glob.glob(pathname)

        images=[]
        for file in files:
            image=cv2.imread(file)
            if image is not None:
                images.append(image)

        boxes,features=get_region_features(images,predictor)
        if boxes is None or features is None:
            logger.warn("物体が一つも検出されませんでした。")
            continue

        title_hash=os.path.basename(directory)
        boxes_save_filepath=os.path.join(boxes_save_dir,title_hash+".pt")
        features_save_filepath=os.path.join(features_save_dir,title_hash+".pt")

        torch.save(boxes,boxes_save_filepath)
        torch.save(features,features_save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--image_root_dir",type=str)
    parser.add_argument("--boxes_save_dir",type=str)
    parser.add_argument("--features_save_dir",type=str)
    parser.add_argument("--index_lower_bound",type=int,default=-1)
    parser.add_argument("--index_upper_bound",type=int,default=-1)
    args=parser.parse_args()

    main(args)
