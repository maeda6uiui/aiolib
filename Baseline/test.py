"""
ベースラインモデルのテストを行う。
"""
import argparse
import logging
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertForMultipleChoice

sys.path.append(".")
import modeling_functions as mf

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    test_input_dir:str=args.test_input_dir
    bert_model_dir:str=args.bert_model_dir
    result_save_dir:str=args.result_save_dir
    test_index_lower_bound:int=args.test_index_lower_bound
    test_index_upper_bound:int=args.test_index_upper_bound

    logger.info("{}からテスト用データローダを作成します。".format(test_input_dir))
    test_dataset=mf.create_dataset(test_input_dir,num_examples=-1,num_options=20)
    test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=False)

    logger.info("{}から事前学習済みの重みを読み込みます。".format(bert_model_dir))
    classifier_model=BertForMultipleChoice.from_pretrained(bert_model_dir)
    classifier_model.to(device)

    for i in range(test_index_lower_bound,test_index_upper_bound):
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(i))
        logger.info("{}からチェックポイントを読み込みます。".format(checkpoint_filepath))
        if os.path.exists(checkpoint_filepath)==False:
            raise RuntimeError("チェックポイントが存在しません。")

        parameters=torch.load(checkpoint_filepath,map_location=device)
        classifier_model.load_state_dict(parameters)

        result_save_filepath=os.path.join(result_save_dir,"result_test_{}.txt".format(i))
        labels_save_filepath=os.path.join(result_save_dir,"labels_test_{}.txt".format(i))
        mf.evaluate_and_save_result(
            classifier_model,
            test_dataloader,
            result_save_filepath,
            labels_save_filepath,
            device,
            logger
        )

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_input_dir",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--result_save_dir",type=str)
    parser.add_argument("--test_index_lower_bound",type=int)
    parser.add_argument("--test_index_upper_bound",type=int)
    args=parser.parse_args()

    main(args)
