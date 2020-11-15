"""
ImageBERTを使用したモデルを訓練する。
"""
import argparse
import logging
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import(
    BertConfig,
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

def main(args):
    train_input_dir:str=args.train_input_dir
    dev_input_dir:str=args.dev_input_dir
    train_qohs_filepath:str=args.train_qohs_filepath
    dev_qohs_filepath:str=args.dev_qohs_filepath
    bert_model_dir:str=args.bert_model_dir
    roi_boxes_dir:str=args.roi_boxes_dir
    roi_features_dir:str=args.roi_features_dir
    max_num_rois:int=args.max_num_rois
    roi_features_dim:int=args.roi_features_dim
    train_batch_size:int=args.train_batch_size
    num_epochs:int=args.num_epochs
    lr:float=args.lr
    result_save_dir:str=args.result_save_dir
    logging_steps:int=args.logging_steps

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

    logger.info("ImageBERTForMultipleChoiceモデルを作成します。")
    config=BertConfig.from_pretrained(bert_model_dir)
    classifier_model=ImageBertForMultipleChoice(config)
    classifier_model.setup_image_bert(bert_model_dir)
    classifier_model.to(device)

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
            device,
            logger,
            logging_steps
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
    parser.add_argument("--roi_boxes_dir",type=str)
    parser.add_argument("--roi_features_dir",type=str)
    parser.add_argument("--max_num_rois",type=int)
    parser.add_argument("--roi_features_dim",type=int)
    parser.add_argument("--train_batch_size",type=int)
    parser.add_argument("--num_epochs",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--result_save_dir",type=str)
    parser.add_argument("--logging_steps",type=int)
    args=parser.parse_args()

    main(args)