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
    def __init__(self,src_dim,dst_dim,seed:int=42):
        set_seed(seed)
        self.fc=torch.nn.Linear(src_dim,dst_dim).to(device)

    def project(self,v:torch.Tensor)->torch.Tensor:
        return self.fc(v)

    def project_from_files(self,src_dir:str,save_dir:str):
        os.makedirs(save_dir,exist_ok=True)

        files=os.listdir(src_dir)
        for file in tqdm(files):
            src_filepath=os.path.join(src_dir,file)
            src_vec=torch.load(src_filepath,map_location=device).to(device)
            dst_vec=self.fc(src_vec)

            save_filepath=os.path.join(save_dir,file)
            torch.save(dst_vec,save_filepath)
