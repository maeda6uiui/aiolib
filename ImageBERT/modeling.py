"""
ImageBERTを使用したモデルを訓練する。
"""
import argparse
import logging
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from transformers import(
    BertConfig,
    BertJapaneseTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

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

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def main(args):
    train_input_dir:str=args.train_input_dir
    dev_input_dir:str=args.dev_input_dir
    train_qohs_filepath:str=args.train_qohs_filepath
    dev_qohs_filepath:str=args.dev_qohs_filepath
    bert_model_dir:str=args.bert_model_dir
    imagebert_checkpoint_filepath:str=args.imagebert_checkpoint_filepath
    roi_boxes_dir:str=args.roi_boxes_dir
    roi_features_dir:str=args.roi_features_dir
    max_num_rois:int=args.max_num_rois
    roi_features_dim:int=args.roi_features_dim
    train_batch_size:int=args.train_batch_size
    num_epochs:int=args.num_epochs
    lr:float=args.lr
    result_save_dir:str=args.result_save_dir
    train_logging_steps:int=args.train_logging_steps
    use_multi_gpus:bool=args.use_multi_gpus
    no_init_from_pretrained_bert:bool=args.no_init_from_pretrained_bert
    use_roi_seq_position:bool=args.use_roi_seq_position

    logger.info("バッチサイズ: {}".format(train_batch_size))
    logger.info("エポック数: {}".format(num_epochs))
    logger.info("学習率: {}".format(lr))

    if use_roi_seq_position:
        logger.info("RoIのSequence Positionに昇順の値を使用します。")

    logger.info("{}から訓練用データセットを作成します。".format(train_input_dir))
    train_dataset=mf.create_dataset(train_input_dir,num_examples=-1,num_options=4)

    logger.info("{}からDev用データローダを作成します。".format(dev_input_dir))
    dev_dataset=mf.create_dataset(dev_input_dir,num_examples=-1,num_options=20)
    dev_dataloader=DataLoader(dev_dataset,batch_size=4,shuffle=False)

    logger.info("問題と選択肢ハッシュ値の辞書を作成します。")
    logger.info("train_qohs_filepath: {}\tdev_qohs_filepath: {}".format(train_qohs_filepath,dev_qohs_filepath))
    train_qohs=mf.load_question_option_hashes(train_qohs_filepath)
    dev_qohs=mf.load_question_option_hashes(dev_qohs_filepath)

    logger.info("RoI情報は以下のディレクトリから読み込まれます。")
    logger.info("roi_boxes_dir: {}\troi_features_dir: {}".format(roi_boxes_dir,roi_features_dir))
    if os.path.exists(roi_boxes_dir)==False:
        logger.warn("roi_boxes_dirは存在しません。")
    if os.path.exists(roi_features_dir)==False:
        logger.warn("roi_features_dirは存在しません。")

    logger.info("ImageBERTForMultipleChoiceモデルを作成します。")
    config=BertConfig.from_pretrained(bert_model_dir)
    classifier_model=ImageBertForMultipleChoice(config)
    if no_init_from_pretrained_bert:
        logger.info("ImageBERTのパラメータを事前学習済みのモデルから初期化しません。")
        tokenizer=BertJapaneseTokenizer.from_pretrained(bert_model_dir)
        classifier_model.imbert.set_sep_token_id(tokenizer.sep_token_id)
    else:
        classifier_model.setup_image_bert(bert_model_dir)
    classifier_model.to(device)

    if imagebert_checkpoint_filepath is not None:
        logger.info("{}からImageBERTのチェックポイントを読み込みます。".format(imagebert_checkpoint_filepath))
        parameters=torch.load(imagebert_checkpoint_filepath,map_location=device)
        parameters=fix_model_state_dict(parameters)
        classifier_model.load_state_dict(parameters,strict=False)

    if use_multi_gpus:
        logger.info("複数のGPUを使用します。")
        classifier_model=nn.DataParallel(classifier_model)
        torch.backends.cudnn.benchmark=True

    num_iterations=len(train_dataset)//train_batch_size
    total_steps=num_iterations*num_epochs

    optimizer=AdamW(classifier_model.parameters(),lr=lr,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    #結果を保存するディレクトリを作成する。
    logger.info("結果は{}に保存されます。".format(result_save_dir))
    os.makedirs(result_save_dir,exist_ok=True)

    #訓練ループ
    for epoch in range(num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch,num_epochs-1))

        #訓練
        train_dataloader=DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
        mean_loss=mf.train(
            classifier_model,
            train_qohs,
            roi_boxes_dir,
            roi_features_dir,
            optimizer,
            scheduler,
            train_dataloader,
            max_num_rois,
            roi_features_dim,
            use_roi_seq_position,
            device,
            logger,
            train_logging_steps
        )
        logger.info("訓練時の損失平均値: {}".format(mean_loss))

        #チェックポイントを保存する。
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch))
        torch.save(classifier_model.state_dict(),checkpoint_filepath)

        #評価
        result_save_filepath=os.path.join(result_save_dir,"result_eval_{}.txt".format(epoch))
        labels_save_filepath=os.path.join(result_save_dir,"labels_eval_{}.txt".format(epoch))
        mf.evaluate_and_save_result(
            classifier_model,
            dev_qohs,
            roi_boxes_dir,
            roi_features_dir,
            dev_dataloader,
            max_num_rois,
            roi_features_dim,
            use_roi_seq_position,
            result_save_filepath,
            labels_save_filepath,
            device,
            logger
        )

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_input_dir",type=str)
    parser.add_argument("--dev_input_dir",type=str)
    parser.add_argument("--train_qohs_filepath",type=str)
    parser.add_argument("--dev_qohs_filepath",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--imagebert_checkpoint_filepath",type=str)
    parser.add_argument("--roi_boxes_dir",type=str)
    parser.add_argument("--roi_features_dir",type=str)
    parser.add_argument("--max_num_rois",type=int)
    parser.add_argument("--roi_features_dim",type=int)
    parser.add_argument("--train_batch_size",type=int)
    parser.add_argument("--num_epochs",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--result_save_dir",type=str)
    parser.add_argument("--train_logging_steps",type=int)
    parser.add_argument("--use_multi_gpus",action="store_true")
    parser.add_argument("--no_init_from_pretrained_bert",action="store_true")
    parser.add_argument("--use_roi_seq_position",action="store_true")
    args=parser.parse_args()

    main(args)
