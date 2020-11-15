import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from transformers import BertForMultipleChoice

def create_dataset(input_dir:str,num_examples:int,num_options:int)->TensorDataset:
    """
    データセットを作成する。
    """
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    input_ids=input_ids[:,:num_options,:]
    attention_mask=attention_mask[:,:num_options,:]
    token_type_ids=token_type_ids[:,:num_options,:]

    if num_examples>0:
        input_ids=input_ids[:num_examples,:,:]
        attention_mask=attention_mask[:num_examples,:,:]
        token_type_ids=token_type_ids[:num_examples,:,:]
        labels=labels[:num_examples]

    return TensorDataset(input_ids,attention_mask,token_type_ids,labels)

def train(
    classifier_model:BertForMultipleChoice,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LambdaLR,
    dataloader:TensorDataset,
    device:torch.device,
    logger:logging.Logger,
    logging_steps:int)->float:
    """
    モデルの訓練を行う。
    """
    classifier_model.train()

    count_steps=0
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
        batch = tuple(t for t in batch)
        bert_inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "token_type_ids": batch[2].to(device),
            "labels": batch[3].to(device)
        }

        classifier_model.zero_grad()
        #Forward propagation
        classifier_outputs=classifier_model(**bert_inputs)
        loss=classifier_outputs[0]
        #Backward propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier_model.parameters(),1.0)
        #Update parameters
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
    classifier_model:BertForMultipleChoice,
    dataloader:DataLoader,
    device:torch.device):
    """
    モデルの評価を行う。
    結果やラベルはDict形式で返される。
    """
    classifier_model.eval()

    count_steps=0
    total_loss=0

    preds=None
    correct_labels=None
    for batch_idx,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        with torch.no_grad():
            batch = tuple(t for t in batch)
            bert_inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "labels": batch[3].to(device)
            }

            classifier_outputs=classifier_model(**bert_inputs)
            loss,logits=classifier_outputs[:2]

            count_steps+=1
            total_loss+=loss.item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                correct_labels = bert_inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                correct_labels = np.append(
                    correct_labels, bert_inputs["labels"].detach().cpu().numpy(), axis=0
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
    classifier_model:BertForMultipleChoice,
    dataloader:DataLoader,
    result_save_filepath:str,
    labels_save_filepath:str,
    device:torch.device,
    logger:logging.Logger):
    """
    モデルの評価を行い、その結果をファイルに保存する。
    """
    res=evaluate(classifier_model,dataloader,device)
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
