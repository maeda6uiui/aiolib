import json
import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from typing import Dict,List,Tuple

sys.path.append(".")
from model import ImageBertForMultipleChoice
import imagebert.utils as imbutil

def load_question_option_hashes(qoh_filepath:str)->Dict[int,List[str]]:
    """
    問題のインデックス(通し番号)とその問題に含まれる選択肢のハッシュ値を読み込む。
    この情報を用いて、各問題の各選択肢に対応するRoI情報を取ってくる。
    """
    with open(qoh_filepath,"r",encoding="utf_8") as r:
        lines=r.read().splitlines()

    qohs={}
    for line in lines:
        data=json.loads(line)

        question_index=data["question_index"]
        option_hashes=data["option_hashes"]

        qohs[question_index]=option_hashes

    return qohs

def create_dataset(input_dir:str,num_examples:int=-1,num_options:int=4)->TensorDataset:
    """
    データセットを作成する。
    """
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    #indicesは各問題に対応するRoIを取ってくるために使用される通し番号
    indices=torch.empty(input_ids.size(0),dtype=torch.long)
    for i in range(input_ids.size(0)):
        indices[i]=i

    input_ids=input_ids[:,:num_options,:]
    attention_mask=attention_mask[:,:num_options,:]
    token_type_ids=token_type_ids[:,:num_options,:]

    if num_examples>0:
        input_ids=input_ids[:num_examples,:,:]
        attention_mask=attention_mask[:num_examples,:,:]
        token_type_ids=token_type_ids[:num_examples,:,:]
        labels=labels[:num_examples]
        indices=indices[:num_examples]

    return TensorDataset(input_ids,attention_mask,token_type_ids,labels,indices)

def create_roi_boxes_and_features(
    qohs:Dict[int,List[str]],
    question_indices:torch.Tensor,  #(N)
    num_options:int,
    roi_boxes_dir:str,
    roi_features_dir:str,
    max_num_rois:int,
    roi_features_dim:int)->Tuple[torch.Tensor,torch.Tensor]:
    """
    ImageBERTのモデルに入力するRoI関連のTensorを作成する。
    """
    batch_size=question_indices.size(0)

    ret_roi_boxes=torch.empty(batch_size,num_options,max_num_rois,4)
    ret_roi_features=torch.empty(batch_size,num_options,max_num_rois,roi_features_dim)

    for i in range(batch_size):
        question_index=question_indices[i]
        qoh=qohs[question_index]

        for j in range(num_options):
            option_hash=qoh[j]
            roi_boxes_filepath=os.path.join(roi_boxes_dir,option_hash+".pt")
            roi_features_filepath=os.path.join(roi_features_dir,option_hash+".pt")

            roi_boxes=imbutil.load_roi_boxes_from_file(roi_boxes_filepath,max_num_rois)
            roi_features=imbutil.load_roi_features_from_file(roi_features_filepath,max_num_rois)
            ret_roi_boxes[i,j]=roi_boxes
            ret_roi_features[i,j]=roi_features

    return ret_roi_boxes,ret_roi_features

def train(
    classifier_model:ImageBertForMultipleChoice,
    qohs:Dict[int,List[str]],
    roi_boxes_dir:str,
    roi_features_dir:str,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LambdaLR,
    dataloader:DataLoader,
    max_num_rois:int,
    roi_features_dim:int,
    device:torch.device,
    logger:logging.Logger,
    logging_steps:int):
    """
    モデルの訓練を行う。
    """
    classifier_model.train()

    count_steps=0
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
        batch = tuple(t for t in batch)

        input_ids=batch[0]
        labels=batch[3]
        question_indices=batch[4]
        num_options=input_ids.size(1)

        roi_boxes,roi_features=create_roi_boxes_and_features(
            qohs,
            question_indices,
            num_options,
            roi_boxes_dir,
            roi_features_dir,
            max_num_rois,
            roi_features_dim
        )

        classifier_inputs={
            "input_ids":input_ids.to(device),
            "roi_boxes":roi_boxes.to(device),
            "roi_features":roi_features.to(device),
            "labels":labels.to(device)
        }

        # Initialize gradiants
        classifier_model.zero_grad()
        # Forward propagation
        outputs = classifier_model(**classifier_inputs)
        loss = outputs[0]
        # Backward propagation
        loss.backward()
        nn.utils.clip_grad_norm_(classifier_model.parameters(), 1.0)
        # Update parameters
        optimizer.step()
        scheduler.step()

        count_steps+=1
        total_loss+=loss.item()

        if batch_idx%logging_steps==0:
            logger.info("Step: {}\tLoss: {}\tlr: {}".format(batch_idx,loss.item(),optimizer.param_groups[0]["lr"]))

    return total_loss/count_steps

def calc_accuracy(pred_labels:np.ndarray, correct_labels:np.ndarray)->float:
    """
    Accuracyを計算する。
    """
    return (pred_labels == correct_labels).mean()

def evaluate(
    classifier_model:ImageBertForMultipleChoice,
    qohs:Dict[int,List[str]],
    roi_boxes_dir:str,
    roi_features_dir:str,
    dataloader:DataLoader,
    max_num_rois:int,
    roi_features_dim:int,
    device:torch.device):
    """
    モデルの評価を行う。
    結果やラベルはDict形式で返される。
    """
    classifier_model.eval()

    count_steps=0
    total_loss=0

    preds = None
    correct_labels = None

    for batch_idx,batch in enumerate(dataloader):
        with torch.no_grad():
            batch = tuple(t for t in batch)
            
            input_ids=batch[0]
            labels=batch[3]
            question_indices=batch[4]
            num_options=input_ids.size(1)
            
            roi_boxes,roi_features=create_roi_boxes_and_features(
                qohs,
                question_indices,
                num_options,
                roi_boxes_dir,
                roi_features_dir,
                max_num_rois,
                roi_features_dim
            )

            classifier_inputs={
                "input_ids":input_ids.to(device),
                "roi_boxes":roi_boxes.to(device),
                "roi_features":roi_features.to(device),
                "labels":labels.to(device)
            }

            outputs = classifier_model(**classifier_inputs)
            loss, logits = outputs[:2]
            
            count_steps+=1
            total_loss+=loss.item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                correct_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                correct_labels = np.append(
                    correct_labels, labels.detach().cpu().numpy(), axis=0
                )

    pred_labels = np.argmax(preds, axis=1)
    accuracy = calc_accuracy(pred_labels, correct_labels)
    eval_loss=total_loss/count_steps

    ret={
        "pred_labels":pred_labels,
        "correct_labels":correct_labels,
        "accuracy":accuracy,
        "eval_loss":eval_loss
    }

    return ret

def evaluate_and_save_result(
    classifier_model:ImageBertForMultipleChoice,
    qohs:Dict[int,List[str]],
    roi_boxes_dir:str,
    roi_features_dir:str,
    dataloader:DataLoader,
    max_num_rois:int,
    roi_features_dim:int,
    result_save_filepath:str,
    labels_save_filepath:str,
    device:torch.device,
    logger:logging.Logger):
    """
    モデルの評価を行い、その結果をファイルに保存する。
    """
    res=evaluate(
        classifier_model,
        qohs,
        roi_boxes_dir,
        roi_features_dir,
        dataloader,
        max_num_rois,
        roi_features_dim,
        device
    )
    accuracy=res["accuracy"]*100.0
    eval_loss=res["eval_loss"]
    logger.info("正解率: {} %".format(accuracy))
    logger.info("評価時の損失平均値: {}".format(eval_loss))

    with open(result_save_filepath,"w",encoding="utf_8",newline="") as w:
        w.write("正解率: {} %\n".format(accuracy))
        w.write("評価時の損失平均値: {}\n".format(eval_loss))

    pred_labels=res["pred_labels"]
    correct_labels=res["correct_labels"]
    with open(labels_save_filepath,"w") as w:
        for pred_label,correct_label in zip(pred_labels,correct_labels):
            w.write("{} {}\n".format(pred_label,correct_label))
