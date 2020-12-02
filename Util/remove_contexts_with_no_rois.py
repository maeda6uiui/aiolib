import argparse
import os
from tqdm import tqdm

def main(context_dir:str,roi_features_dir:str):
    files=os.listdir(context_dir)
    count_removals=0

    for file in tqdm(files):
        #RoI情報の存在しないコンテキストを削除する。
        roi_features_filepath=os.path.join(roi_features_dir,file)
        if os.path.exists(roi_features_filepath)==False:
            context_filepath=os.path.join(context_dir,file)
            os.remove(context_filepath)
            count_removals+=1

    print(count_removals)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--context_dir",type=str)
    parser.add_argument("--roi_features_dir",type=str)
    args=parser.parse_args()

    main(args.context_dir,args.roi_features_dir)
