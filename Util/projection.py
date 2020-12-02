import argparse
import os
import torch
from tqdm import tqdm

def main(
    cls_predictor_filepath:str,
    input_dir:str,
    save_dir:str):
    os.makedirs(save_dir,exist_ok=True)

    device=torch.device("cpu")

    cls_predictor=torch.nn.Linear(1024,81)
    cls_predictor_parameters=torch.load(cls_predictor_filepath,map_location=device)
    cls_predictor.load_state_dict(cls_predictor_parameters)

    files=os.listdir(input_dir)
    for file in tqdm(files):
        input_filepath=os.path.join(input_dir,file)
        roi_features=torch.load(input_filepath,map_location=device)
        if roi_features.size(0)==0:
            continue

        cls_pred=cls_predictor(roi_features)
        cls_pred=torch.argmax(cls_pred,dim=1)

        save_filepath=os.path.join(save_dir,file)
        torch.save(cls_pred,save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cls_predictor_filepath",type=str)
    parser.add_argument("--input_dir",type=str)
    parser.add_argument("--save_dir",type=str)
    args=parser.parse_args()

    main(
        args.cls_predictor_filepath,
        args.input_dir,
        args.save_dir
    )
