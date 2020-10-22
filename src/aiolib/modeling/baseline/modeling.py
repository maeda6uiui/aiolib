import argparse
import logging
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import(
    Dataset,
    DataLoader,
    TensorDataset
)
from transformers import(
    BertForMultipleChoice,
    AdamW,
    get_linear_schedule_with_warmup
)

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed:int):
    """
    乱数のシードを設定する。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_dataset(input_dir:str,num_examples:int=-1,num_options:int=4)->TensorDataset:
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
    logger:logging.Logger,
    logging_steps:int=100)->float:
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

def simple_accuracy(pred_labels:np.ndarray, correct_labels:np.ndarray):
    """
    Accuracyを計算する。
    """
    return (pred_labels == correct_labels).mean()

def evaluate(classifier_model:BertForMultipleChoice,dataloader:DataLoader):
    """
    モデルの評価を行う。
    結果やラベルはDict形式で返される。
    """
    classifier_model.eval()

    preds=None
    correct_labels=None
    count_steps=0
    total_eval_loss=0
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
            total_eval_loss+=loss.item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                correct_labels = bert_inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                correct_labels = np.append(
                    correct_labels, bert_inputs["labels"].detach().cpu().numpy(), axis=0
                )

    pred_labels = np.argmax(preds, axis=1)
    accuracy = simple_accuracy(pred_labels, correct_labels)
    eval_loss=total_eval_loss/count_steps

    ret={
        "pred_labels":pred_labels,
        "correct_labels":correct_labels,
        "accuracy":accuracy,
        "eval_loss":eval_loss
    }

    return ret

class BaselineModeler(object):
    """
    ベースラインモデルを訓練する。
    """
    def __init__(
        self,
        train_input_dir:str,
        dev_input_dir:str,
        bert_model_dir:str,
        seed:int=42,
        logger:logging.Logger=default_logger):
        logger.info("{}から訓練用データセットを作成します。".format(train_input_dir))
        self.train_dataset=create_dataset(train_input_dir,num_examples=-1,num_options=4)

        logger.info("{}からDev用データローダを作成します。".format(dev_input_dir))
        dev_dataset=create_dataset(dev_input_dir,num_examples=-1,num_options=20)
        self.dev_dataloader=DataLoader(dev_dataset,batch_size=4,shuffle=False)

        self.bert_model_dir=bert_model_dir
        self.__create_classifier_model(bert_model_dir)

        logger.info("シード: {}".format(seed))
        set_seed(seed)

        self.logger=logger

    def __create_classifier_model(self,bert_model_dir:str):
        self.classifier_model=None
        if bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERT Pre-trainedモデルを読み込みます。")
            self.classifier_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込みます。".format(bert_model_dir))
            self.classifier_model=BertForMultipleChoice.from_pretrained(bert_model_dir)
        self.classifier_model.to(device)

    def train_and_eval(
        self,
        train_batch_size:int=4,
        num_epochs:int=5,
        lr:float=2.5e-5,
        init_parameters=False,
        result_save_dir:str="./OutputDir",
        logging_steps:int=100):
        logger=self.logger
        logger.info("モデルの訓練を開始します。")
        logger.info("バッチサイズ: {}".format(train_batch_size))
        logger.info("エポック数: {}".format(num_epochs))
        logger.info("学習率: {}".format(lr))

        if init_parameters:
            self.__create_classifier_model(self.bert_model_dir)

        num_iterations=len(self.train_dataset)//train_batch_size
        total_steps=num_iterations*num_epochs

        optimizer=AdamW(self.classifier_model.parameters(),lr=lr,eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        #結果を保存するディレクトリを作成する。
        logger.info("結果は{}に保存されます。".format(result_save_dir))
        os.makedirs(result_save_dir,exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info("===== Epoch {}/{} =====".format(epoch+1,num_epochs))

            #訓練
            train_dataloader=DataLoader(self.train_dataset,batch_size=train_batch_size,shuffle=True)
            mean_loss=train(
                self.classifier_model,
                optimizer,
                scheduler,
                train_dataloader,
                logger,
                logging_steps=logging_steps)
            logger.info("訓練時の損失平均値: {}".format(mean_loss))

            #チェックポイントを保存する。
            checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
            torch.save(self.classifier_model.state_dict(),checkpoint_filepath)

            #評価
            res=evaluate(self.classifier_model,self.dev_dataloader)
            accuracy=res["accuracy"]*100.0
            eval_loss=res["eval_loss"]
            logger.info("正解率: {} %".format(accuracy))
            logger.info("評価時の損失平均値: {}".format(eval_loss))

            #テキストファイルに評価の結果を保存する。
            result_filepath=os.path.join(result_save_dir,"result_eval_{}.txt".format(epoch+1))
            labels_filepath=os.path.join(result_save_dir,"labels_eval_{}.txt".format(epoch+1))

            with open(result_filepath,"w",encoding="utf_8",newline="") as w:
                w.write("正解率: {} %\n".format(accuracy))
                w.write("評価時の損失平均値: {}\n".format(eval_loss))

            pred_labels=res["pred_labels"]
            correct_labels=res["correct_labels"]
            with open(labels_filepath,"w") as w:
                for pred_label,correct_label in zip(pred_labels,correct_labels):
                    w.write("{} {}\n".format(pred_label,correct_label))

class BaselineTester(object):
    """
    ベースラインモデルのテストを行う。
    """
    def __init__(
        self,
        test_input_dir:str,
        bert_model_dir:str,
        seed:int=42,
        logger:logging.Logger=default_logger):
        logger.info("{}からテスト用データローダを作成します。".format(test_input_dir))
        test_dataset=create_dataset(test_input_dir,num_examples=-1,num_options=20)
        self.test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=False)

        self.classifier_model=None
        if bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERT Pre-trainedモデルを読み込みます。")
            self.classifier_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込みます。".format(bert_model_dir))
            self.classifier_model=BertForMultipleChoice.from_pretrained(bert_model_dir)
        self.classifier_model.to(device)

        logger.info("シード: {}".format(seed))
        set_seed(seed)

        self.logger=logger

    def test(
        self,
        model_filepath:str,
        result_filepath:str,
        labels_filepath:str):
        logger=self.logger
        logger.info("モデルのテストを開始します。")

        #モデルのパラメータを読み込む。
        logger.info("{}からモデルパラメータを読み込みます。".format(model_filepath))
        parameters=torch.load(model_filepath,map_location=device)
        self.classifier_model.load_state_dict(parameters)

        #評価
        res=evaluate(self.classifier_model,self.test_dataloader)
        accuracy=res["accuracy"]*100.0
        eval_loss=res["eval_loss"]
        logger.info("正解率: {} %".format(accuracy))
        logger.info("評価時の損失平均値: {}".format(eval_loss))

        #テキストファイルに評価の結果を保存する。
        with open(result_filepath,"w",encoding="utf_8",newline="") as w:
            w.write("正解率: {} %\n".format(accuracy))
            w.write("評価時の損失平均値: {}\n".format(eval_loss))

        pred_labels=res["pred_labels"]
        correct_labels=res["correct_labels"]
        with open(labels_filepath,"w") as w:
            for pred_label,correct_label in zip(pred_labels,correct_labels):
                w.write("{} {}\n".format(pred_label,correct_label))
