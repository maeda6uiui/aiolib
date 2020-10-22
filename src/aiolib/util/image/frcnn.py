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

from typing import List

from .. import hashing

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_region_features_single(raw_image:np.ndarray,predictor:DefaultPredictor)->torch.Tensor:
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
        box_features=model.roi_heads.box_pooler(
            [features[f] for f in features if f!="p6"],
            [x.pred_boxes for x in instances]
        )
        box_features=model.roi_heads.box_head(box_features) #FC層の出力

        return box_features #(RoIの数,特徴量の次元数)

def get_region_features(raw_images:List[np.ndarray],predictor:DefaultPredictor)->torch.Tensor:
    """
    複数の画像から特徴量を取得する。
    """
    features_list=[]
    for raw_image in raw_images:
        features=get_region_features_single(raw_image,predictor)
        features_list.append(features)

    #Tensorを結合する。
    if len(features_list)==0:
        return torch.zeros(0,0).to(device)

    dimension=features_list[0].size(1)
    ret=torch.empty(0,dimension).to(device)
    for features in features_list:
        ret=torch.cat([ret,features],dim=0)
    
    return ret

class ImageFeatureExtractorBase(object):
    """
    基底クラス
    """
    def __init__(self,model_name:str):
        cfg=get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
        cfg.MODEL.DEVICE="cuda" if torch.cuda.is_available() else "cpu"
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
        logger:logging.Logger=default_logger):
        super().__init__(model_name)
        self.articles_list=pd.read_csv(article_list_filepath,encoding="utf_8",sep="\t")
        self.logger=logger

    def extract(self,image_root_dir:str,save_dir:str):
        """
        画像の特徴量を抽出する。
        """
        os.makedirs(save_dir,exist_ok=True)

        for row in tqdm(self.articles_list.values):
            article_name,sec1,sec2=row[:3]
            image_dir=os.path.join(image_root_dir,str(sec1),str(sec2))

            pathname=os.path.join(image_dir,"*")
            files=glob.glob(pathname)
            images=[]
            for file in files:
                image=cv2.imread(file)
                images.append(image)

            features=get_region_features(images,self.predictor)
            
            title_hash=hashing.get_md5_hash(article_name)
            save_filepath=os.path.join(save_dir,title_hash+".pt")
            torch.save(features,save_filepath)

class ImageFeatureExtractor(ImageFeatureExtractorBase):
    """
    検索エンジンから収集された画像を使用して画像の特徴量を抽出する。
    """
    def __init__(
        self,
        model_name:str="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        logger:logging.Logger=default_logger):
        super().__init__(model_name)
        self.logger=logger

    def extract(self,image_root_dir:str,save_dir:str):
        """
        画像の特徴量を抽出する。
        """
        os.makedirs(save_dir,exist_ok=True)

        pathname=os.path.join(image_root_dir,"*")
        directories=glob.glob(pathname)
        for directory in tqdm(directories,total=len(directories)):
            pathname=os.path.join(directory,"*[!txt]")
            files=glob.glob(pathname)

            images=[]
            for file in files:
                image=cv2.imread(file)
                images.append(image)

            features=get_region_features(images,self.predictor)

            title_hash=os.path.basename(directory)
            save_filepath=os.path.join(save_dir,title_hash+".pt")
            torch.save(features,save_filepath)
