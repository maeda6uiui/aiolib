"""
ImageBERTを使用したモデルをテストする。
"""
import argparse
import logging
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig

sys.path.append(".")
import modeling_functions as mf
from model import ImageBertForMultipleChoice

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#シードを設定する。
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True

def main(args):
    test_input_dir:str=args.test_input_dir
    test_qohs_filepath:str=args.test_qohs_filepath
    bert_model_dir:str=args.bert_model_dir
    imagebert_checkpoint_filepath:str=args.imagebert_checkpoint_filepath
    roi_boxes_dir:str=args.roi_boxes_dir
    roi_features_dir:str=args.roi_features_dir
    max_num_rois:int=args.max_num_rois
    roi_features_dim:int=args.roi_features_dim
    result_save_dir:str=args.result_save_dir
    test_index_lower_bound:int=args.test_index_lower_bound
    test_index_upper_bound:int=args.test_index_upper_bound
    use_roi_seq_position:bool=args.use_roi_seq_position

    if use_roi_seq_position:
        logger.info("RoIのSequence Positionに昇順の値を使用します。")

    logger.info("{}からテスト用データローダを作成します。".format(test_input_dir))
    test_dataset=mf.create_dataset(test_input_dir,num_examples=-1,num_options=20)
    test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=False)

    logger.info("問題と選択肢ハッシュ値の辞書を作成します。")
    logger.info("test_qohs_filepath: {}".format(test_qohs_filepath))
    test_qohs=mf.load_question_option_hashes(test_qohs_filepath)

    logger.info("RoI情報は以下のディレクトリから読み込まれます。")
    logger.info("roi_boxes_dir: {}\troi_features_dir: {}".format(roi_boxes_dir,roi_features_dir))
    if os.path.exists(roi_boxes_dir)==False:
        logger.warn("roi_boxes_dirは存在しません。")
    if os.path.exists(roi_features_dir)==False:
        logger.warn("roi_features_dirは存在しません。")

    logger.info("ImageBERTForMultipleChoiceモデルを作成します。")
    config=BertConfig.from_pretrained(bert_model_dir)
    classifier_model=ImageBertForMultipleChoice(config)
    classifier_model.setup_image_bert(bert_model_dir)
    classifier_model.to(device)

    if imagebert_checkpoint_filepath is not None:
        logger.info("{}からImageBERTのチェックポイントを読み込みます。".format(imagebert_checkpoint_filepath))
        parameters=torch.load(imagebert_checkpoint_filepath,map_location=device)
        classifier_model.load_state_dict(parameters,strict=False)

    for i in range(test_index_lower_bound,test_index_upper_bound):
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(i))
        logger.info("{}からチェックポイントを読み込みます。".format(checkpoint_filepath))
        if os.path.exists(checkpoint_filepath)==False:
            raise RuntimeError("チェックポイントが存在しません。")

        #評価
        result_save_filepath=os.path.join(result_save_dir,"result_test_{}.txt".format(i))
        labels_save_filepath=os.path.join(result_save_dir,"labels_test_{}.txt".format(i))
        logits_save_filepath=os.path.join(result_save_dir,"logits_test_{}.txt".format(i))
        mf.evaluate_and_save_result(
            classifier_model,
            test_qohs,
            roi_boxes_dir,
            roi_features_dir,
            test_dataloader,
            max_num_rois,
            roi_features_dim,
            use_roi_seq_position,
            result_save_filepath,
            labels_save_filepath,
            logits_save_filepath,
            device,
            logger
        )

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_input_dir",type=str)
    parser.add_argument("--test_qohs_filepath",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--imagebert_checkpoint_filepath",type=str)
    parser.add_argument("--roi_boxes_dir",type=str)
    parser.add_argument("--roi_features_dir",type=str)
    parser.add_argument("--max_num_rois",type=int)
    parser.add_argument("--roi_features_dim",type=int)
    parser.add_argument("--result_save_dir",type=str)
    parser.add_argument("--test_index_lower_bound",type=int)
    parser.add_argument("--test_index_upper_bound",type=int)
    parser.add_argument("--use_roi_seq_position",action="store_true")
    args=parser.parse_args()

    main(args)
