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
        seed:int=42,
        logger:logging.Logger=default_logger):
        set_seed(seed)
        
        self.fc_boxes=torch.nn.Linear(4,dst_dim).to(device)
        self.fc_features=torch.nn.Linear(features_src_dim,dst_dim).to(device)

        self.logger=logger

    def project_boxes(self,v:torch.Tensor)->torch.Tensor:
        return self.fc_boxes(v)
    def project_features(self,v:torch.Tensor)->torch.Tensor:
        return self.fc_features(v)

    def project_from_directory(
        self,
        boxes_src_dir:str,
        boxes_save_dir:str,
        features_src_dir:str,
        features_save_dir:str):
        logger=self.logger

        os.makedirs(boxes_save_dir,exist_ok=True)
        os.makedirs(features_save_dir,exist_ok=True)

        #Boxes
        logger.info("RoI座標の射影を開始します。")

        files=os.listdir(boxes_src_dir)
        for file in tqdm(files):
            src_filepath=os.path.join(boxes_src_dir,file)
            src_vec=torch.load(src_filepath,map_location=device).to(device)
            dst_vec=self.fc_boxes(src_vec)

            save_filepath=os.path.join(boxes_save_dir,file)
            torch.save(dst_vec,save_filepath)

        #Features
        logger.info("RoI特徴量の射影を開始します。")

        files=os.listdir(features_src_dir)
        for file in tqdm(files):
            src_filepath=os.path.join(features_src_dir,file)
            src_vec=torch.load(src_filepath,map_location=device).to(device)
            dst_vec=self.fc_features(src_vec)

            save_filepath=os.path.join(features_save_dir,file)
            torch.save(dst_vec,save_filepath)
