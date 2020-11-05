import cv2
import glob
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import torch
from tqdm import tqdm

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

from typing import List,Tuple

from .. import hashing

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

default_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_region_features_single(
    raw_image:np.ndarray,
    predictor:DefaultPredictor,
    device:torch.device=default_device)->Tuple[torch.Tensor,torch.Tensor]:
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
    predictor:DefaultPredictor,
    device:torch.device=default_device)->Tuple[torch.Tensor,torch.Tensor]:
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

class ImageFeatureExtractorBase(object):
    """
    基底クラス
    """
    def __init__(self,model_name:str,device:torch.device=default_device):
        cfg=get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
        cfg.MODEL.DEVICE=str(device)
        cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url(model_name)
        self.predictor=DefaultPredictor(cfg)

class WikipediaImageFeatureExtractor(ImageFeatureExtractorBase):
    """
    Wikipediaから収集された画像を使用して画像の特徴量を抽出する。
    """
    def __init__(
        self,
        article_list_filepath:str,
        model_name:str="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        device:torch.device=default_device,
        logger:logging.Logger=default_logger):
        super().__init__(model_name,device=device)
        self.articles_list=pd.read_csv(article_list_filepath,encoding="utf_8",sep="\t")
        self.logger=logger

    def extract(self,image_root_dir:str,boxes_save_dir:str,features_save_dir:str):
        """
        画像の特徴量を抽出する。
        """
        logger=self.logger

        os.makedirs(boxes_save_dir,exist_ok=True)
        os.makedirs(features_save_dir,exist_ok=True)

        for row in tqdm(self.articles_list.values):
            article_name,sec1,sec2=row[:3]
            image_dir=os.path.join(image_root_dir,str(sec1),str(sec2))

            pathname=os.path.join(image_dir,"*")
            files=glob.glob(pathname)
            images=[]
            for file in files:
                image=cv2.imread(file)
                if image is not None:
                    images.append(image)

            boxes,features=get_region_features(images,self.predictor)
            if boxes is None or features is None:
                logger.warn("物体が一つも検出されませんでした。 記事名: {}".format(article_name))
                continue
            
            title_hash=hashing.get_md5_hash(article_name)
            boxes_save_filepath=os.path.join(boxes_save_dir,title_hash+".pt")
            features_save_filepath=os.path.join(features_save_dir,title_hash+".pt")

            torch.save(boxes,boxes_save_filepath)
            torch.save(features,features_save_filepath)

class ImageFeatureExtractor(ImageFeatureExtractorBase):
    """
    検索エンジンから収集された画像を使用して画像の特徴量を抽出する。
    """
    def __init__(
        self,
        model_name:str="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        device:torch.device=default_device,
        logger:logging.Logger=default_logger):
        super().__init__(model_name,device=device)
        self.logger=logger

    def extract(
        self,
        image_root_dir:str,
        boxes_save_dir:str,
        features_save_dir:str,
        index_lower_bound:int=-1,
        index_upper_bound:int=-1):
        """
        画像の特徴量を抽出する。
        """
        logger=self.logger

        os.makedirs(boxes_save_dir,exist_ok=True)
        os.makedirs(features_save_dir,exist_ok=True)

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

            boxes,features=get_region_features(images,self.predictor)
            if boxes is None or features is None:
                logger.warn("物体が一つも検出されませんでした。")
                continue

            title_hash=os.path.basename(directory)
            boxes_save_filepath=os.path.join(boxes_save_dir,title_hash+".pt")
            features_save_filepath=os.path.join(features_save_dir,title_hash+".pt")

            torch.save(boxes,boxes_save_filepath)
            torch.save(features,features_save_filepath)
