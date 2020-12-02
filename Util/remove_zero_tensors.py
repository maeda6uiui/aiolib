import argparse
import glob
import torch
import os
from tqdm import tqdm

def main(input_dir:str):
    device=torch.device("cpu")

    pathname=os.path.join(input_dir,"*")
    files=glob.glob(pathname)
    count_removals=0
    for file in tqdm(files):
        tensor=torch.load(file,map_location=device)
        if tensor.size(0)==0:
            os.remove(file)
            count_removals+=1

    print(count_removals)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str)
    args=parser.parse_args()

    main(args.input_dir)
