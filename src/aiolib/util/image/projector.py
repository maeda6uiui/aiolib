"""
特徴量ベクトルの射影を行う。
"""
import argparse
import logging
import numpy as np
import os
import random
import torch
from tqdm import tqdm

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Projector(object):
    """
    特徴量ベクトルの射影を行う。
    """
    def __init__(
        self,
        features_src_dim:int,
        dst_dim:int,
        image_width:int=256,
        image_height:int=256,
        seed:int=42,
        logger:logging.Logger=default_logger):
        self.logger=logger

        set_seed(seed)
        
        self.fc_boxes=torch.nn.Linear(5,dst_dim).to(device)
        self.fc_features=torch.nn.Linear(features_src_dim,dst_dim).to(device)

        self.image_width=image_width
        self.image_height=image_height

    def project_boxes(self,src_vec:torch.Tensor)->torch.Tensor:
        num_rois=src_vec.size(0)
        pos_vec=torch.empty(num_rois,5).to(device)

        for i in range(num_rois):
            x_tl=src_vec[i,0]
            y_tl=src_vec[i,1]
            x_br=src_vec[i,2]
            y_br=src_vec[i,3]

            pos_vec[i,0]=x_tl/self.image_width
            pos_vec[i,1]=y_tl/self.image_height
            pos_vec[i,2]=x_br/self.image_width
            pos_vec[i,3]=y_br/self.image_height
            pos_vec[i,4]=(x_br-x_tl)*(y_br-y_tl)/(self.image_width*self.image_height)

        return self.fc_boxes(pos_vec)
        
    def project_features(self,src_vec:torch.Tensor)->torch.Tensor:
        return self.fc_features(src_vec)

    def project_from_directory(
        self,
        boxes_src_dir:str,
        boxes_save_dir:str,
        features_src_dir:str,
        features_save_dir:str):
        logger=self.logger

        #保存先ディレクトリがすでに存在する場合には失敗する。
        os.makedirs(boxes_save_dir)
        os.makedirs(features_save_dir)

        #Boxes
        logger.info("RoI座標の射影を開始します。")

        files=os.listdir(boxes_src_dir)
        for file in tqdm(files):
            src_filepath=os.path.join(boxes_src_dir,file)
            src_vec=torch.load(src_filepath,map_location=device).to(device) #(RoIの数,4)
            dst_vec=self.project_boxes(src_vec)

            save_filepath=os.path.join(boxes_save_dir,file)
            torch.save(dst_vec,save_filepath)

        #Features
        logger.info("RoI特徴量の射影を開始します。")

        files=os.listdir(features_src_dir)
        for file in tqdm(files):
            src_filepath=os.path.join(features_src_dir,file)
            src_vec=torch.load(src_filepath,map_location=device).to(device)
            dst_vec=self.project_features(src_vec)

            save_filepath=os.path.join(features_save_dir,file)
            torch.save(dst_vec,save_filepath)
