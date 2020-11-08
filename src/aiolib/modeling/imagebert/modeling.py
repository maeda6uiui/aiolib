import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from transformers import(
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup
)

from typing import List,Tuple

from .models import ImageBertForMultipleChoice

sys.path.append("../../")
from util import hashing

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

default_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuestionOptions(object):
    """
    問題の選択肢
    """
    def __init__(self):
        self.options=[]

    def append(self,option:str):
        self.options.append(option)

    def get(self,index:int):
        return self.options[index]

def load_options_list(list_filepath:str,logger:logging.Logger=default_logger)->List[QuestionOptions]:
    logger.info("{}から選択肢のリストを読み込みます。".format(list_filepath))

    with open(list_filepath,"r",encoding="UTF-8") as r:
        lines=r.read().splitlines()

    options=[]
    ops=None
    for line in lines:
        if ops is None:
            ops=QuestionOptions()

        if line=="":
            options.append(ops)
            ops=None
        else:
            ops.append(line)

    return options

def create_dataset(input_dir:str,num_examples:int=-1,num_options:int=4)->TensorDataset:
    """
    データセットを作成する。
    """
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    #indicesは各問題に対応する画像を取ってくるために使用する。
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

    return TensorDataset(indices,input_ids,attention_mask,token_type_ids,labels)

def __get_roi_boxes_and_features(
    options:QuestionOptions,
    num_options:int,
    roi_boxes_dir:str,
    roi_features_dir:str,
    device:torch.device=default_device)->Tuple[List[torch.Tensor],List[torch.Tensor]]:
    """
    各選択肢に対応するRoIの矩形領域情報をファイルから読み込む。

    出力Tensorのサイズ
    (num_rois,roi_features_dim)
    """
    roi_boxes_list=[]
    roi_features_list=[]

    for i in range(num_options):
        option=options.get(i)
        title_hash=hashing.get_md5_hash(option)

        roi_boxes_filepath=os.path.join(roi_boxes_dir,title_hash+".pt")
        roi_features_filepath=os.path.join(roi_features_dir,title_hash+".pt")

        #画像の特徴量が存在する場合 (矩形領域の座標データも存在するはず)
        if os.path.exists(roi_features_filepath):
            roi_boxes=torch.load(roi_boxes_filepath,map_location=device).to(device)
            roi_features=torch.load(roi_features_filepath,map_location=device).to(device)
        else:
            roi_boxes=None
            roi_features=None

        roi_boxes_list.append(roi_boxes)
        roi_features_list.append(roi_features)

    return roi_boxes_list,roi_features_list

def __trim_roi_tensors(
    tensors:List[torch.Tensor],
    max_num_rois:int,
    out_dim:int,
    device:torch.device=default_device)->torch.Tensor:
    """
    各バッチの各選択肢で含まれるRoIの数が異なると処理が面倒なので、max_num_roisに合わせる。
    もしもmax_num_roisよりも多い場合には切り捨て、max_num_roisよりも少ない場合には0ベクトルで埋める。

    入力Tensorのサイズ
    [(num_rois,out_dim)]    num_options個のTensor

    出力Tensorのサイズ
    (num_options,max_num_rois,out_dim)
    """
    num_options=len(tensors)
    ret=torch.empty(num_options,max_num_rois,out_dim).to(device)

    for i,tensor in enumerate(tensors):
        #選択肢に対応するRoIが存在しない場合
        if tensor is None:
            ret[i]=torch.zeros(max_num_rois,out_dim).to(device)
            continue

        num_rois=tensor.size(0)

        #RoIの数が制限よりも多い場合はTruncateする。
        if num_rois>max_num_rois:
            tensor=tensor[:max_num_rois]
        #RoIの数が制限よりも少ない場合は0ベクトルで埋める。
        elif num_rois<max_num_rois:
            zeros=torch.zeros(max_num_rois-num_rois,out_dim).to(device)
            tensor=torch.cat([tensor,zeros],dim=0)

        ret[i]=tensor

    return ret

def create_roi_boxes_and_tensors(
    options_list:List[QuestionOptions], #N個のQuestionOptionsを含むList
    question_indices:torch.Tensor,  #問題のIndex (N個)
    num_options:int,    #選択肢の数
    roi_boxes_dir:str,
    roi_features_dir:str,
    max_num_rois:int,
    roi_features_dim:int,
    device:torch.device=default_device)->Tuple[torch.Tensor,torch.Tensor]:
    """
    ImageBERTのモデルに入力するためのRoI特徴量を作成する。
    """
    batch_size=question_indices.size(0)

    ret_roi_boxes=torch.empty(batch_size,num_options,max_num_rois,4).to(device)
    ret_roi_features=torch.empty(batch_size,num_options,max_num_rois,roi_features_dim).to(device)

    for i in range(batch_size):
        options=options_list[question_indices[i]]
        #RoIの座標情報と特徴量をファイルから読み込む。
        roi_boxes_list,roi_features_list=__get_roi_boxes_and_features(
            options,num_options,roi_boxes_dir,roi_features_dir,device=device
        )
        #選択肢ごとに含まれるRoIの数が異なると処理が面倒なので、max_num_roisに合わせる。
        roi_boxes=__trim_roi_tensors(roi_boxes_list,max_num_rois,4,device=device)
        roi_features=__trim_roi_tensors(roi_features_list,max_num_rois,roi_features_dim,device=device)

        ret_roi_boxes[i]=roi_boxes
        ret_roi_features[i]=roi_features

    return ret_roi_boxes,ret_roi_features

def train(
    classifier_model:ImageBertForMultipleChoice,
    options_list:List[QuestionOptions],
    roi_boxes_dir:str,
    roi_features_dir:str,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LambdaLR,
    dataloader:DataLoader,
    max_num_rois:int=100,
    roi_features_dim:int=1024,
    device:torch.device=default_device,
    logger:logging.Logger=default_logger,
    logging_steps:int=100):
    """
    モデルの訓練を行う。
    """
    classifier_model.train()

    count_steps=0
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
        batch = tuple(t for t in batch)
        bert_inputs = {
            "indices":batch[0].to(device),
            "input_ids": batch[1].to(device),
            "labels": batch[4].to(device)
        }
        num_options=bert_inputs["input_ids"].size(1)
        
        roi_boxes,roi_features=create_roi_boxes_and_tensors(
            options_list,
            bert_inputs["indices"],
            num_options,
            roi_boxes_dir,
            roi_features_dir,
            max_num_rois,
            roi_features_dim,
            device=device
        )

        classifier_inputs={
            "input_ids":bert_inputs["input_ids"],
            "roi_boxes":roi_boxes,
            "roi_features":roi_features,
            "labels":bert_inputs["labels"],
            "max_num_rois":max_num_rois
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

def simple_accuracy(pred_labels:np.ndarray, correct_labels:np.ndarray)->float:
    """
    Accuracyを計算する。
    """
    return (pred_labels == correct_labels).mean()

def evaluate(
    classifier_model:ImageBertForMultipleChoice,
    options_list:List[QuestionOptions],
    roi_boxes_dir:str,
    roi_features_dir:str,
    dataloader:DataLoader,
    max_num_rois:int=100,
    roi_features_dim:int=1024,
    device:torch.device=default_device):
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
            bert_inputs = {
                "indices":batch[0].to(device),
                "input_ids": batch[1].to(device),
                "labels": batch[4].to(device)
            }
            num_options=bert_inputs["input_ids"].size(1)
        
            roi_boxes,roi_features=create_roi_boxes_and_tensors(
                options_list,
                bert_inputs["indices"],
                num_options,
                roi_boxes_dir,
                roi_features_dir,
                max_num_rois,
                roi_features_dim,
                device=device
            )

            classifier_inputs={
                "input_ids":bert_inputs["input_ids"],
                "roi_boxes":roi_boxes,
                "roi_features":roi_features,
                "labels":bert_inputs["labels"],
                "max_num_rois":max_num_rois
            }

            outputs = classifier_model(**classifier_inputs)
            loss, logits = outputs[:2]
            
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
    accuracy = simple_accuracy(pred_labels, correct_labels)
    eval_loss=total_loss/count_steps

    ret={
        "pred_labels":pred_labels,
        "correct_labels":correct_labels,
        "accuracy":accuracy,
        "eval_loss":eval_loss
    }

    return ret

class ImageBertModeler(object):
    """
    画像の特徴量を使用してモデルのFine-Tuningを行う。
    """
    def __init__(
        self,
        train_input_dir:str,
        dev_input_dir:str,
        bert_model_dir:str,
        roi_boxes_dir:str,
        roi_features_dir:str,
        max_num_rois:int=100,
        logger:logging.Logger=default_logger):
        self.logger=logger

        logger.info("{}から訓練用データセットを作成します。".format(train_input_dir))
        self.train_dataset=create_dataset(train_input_dir,num_examples=-1,num_options=4)

        logger.info("{}からDev用データローダを作成します。".format(dev_input_dir))
        dev_dataset=create_dataset(dev_input_dir,num_examples=-1,num_options=20)
        self.dev_dataloader=DataLoader(dev_dataset,batch_size=4,shuffle=False)

        logger.info("選択肢のリストを読み込みます。")
        self.train_options=load_options_list(os.path.join(train_input_dir,"options_list.txt"))
        self.dev_options=load_options_list(os.path.join(dev_input_dir,"options_list.txt"))

        logger.info("roi_boxes_dir: {}\troi_features_dir: {}".format(roi_boxes_dir,roi_features_dir))
        self.roi_boxes_dir=roi_boxes_dir
        self.roi_features_dir=roi_features_dir
        self.max_num_rois=max_num_rois

        self.bert_model_dir=bert_model_dir
        self.__create_classifier_model()

        self.device=torch.device("cpu") #デフォルトではCPU

    def __create_classifier_model(self):
        logger=self.logger

        self.classifier_model=None
        if self.bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERTモデルを用いて分類器のパラメータを初期化します。")
            config=BertConfig.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
            self.classifier_model=ImageBertForMultipleChoice(config)
            self.classifier_model.initialize_from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込んで分類器のパラメータを初期化します。".format(self.bert_model_dir))
            config=BertConfig.from_json_file(self.bert_model_dir)
            self.classifier_model=ImageBertForMultipleChoice(config)
            self.classifier_model.initialize_from_pretrained(self.bert_model_dir)

    def to(self,device:torch.device):
        self.device=device
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
                self.train_options,
                self.roi_boxes_dir,
                self.roi_features_dir,
                optimizer,
                scheduler,
                train_dataloader,
                max_num_rois=self.max_num_rois,
                device=self.device,
                logger=logger,
                logging_steps=logging_steps
            )
            logger.info("訓練時の損失平均値: {}".format(mean_loss))

            #チェックポイントを保存する。
            checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
            torch.save(self.classifier_model.state_dict(),checkpoint_filepath)

            #評価
            res=evaluate(
                self.classifier_model,
                self.dev_options,
                self.roi_boxes_dir,
                self.roi_features_dir,
                self.dev_dataloader,
                max_num_rois=self.max_num_rois,
                device=self.device
            )
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

class ImageBertTester(object):
    """
    画像の特徴量を使用してFine-Tuningされたモデルのテストを行う。
    """
    def __init__(
        self,
        test_input_dir:str,
        bert_model_dir:str,
        roi_boxes_dir:str,
        roi_features_dir:str,
        max_num_rois:int=100,
        logger:logging.Logger=default_logger):
        self.logger=logger

        logger.info("{}からテスト用データローダを作成します。".format(test_input_dir))
        test_dataset=create_dataset(test_input_dir,num_examples=-1,num_options=20)
        self.test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=False)

        logger.info("選択肢のリストを読み込みます。")
        self.test_options=load_options_list(os.path.join(test_input_dir,"options_list.txt"))

        logger.info("roi_boxes_dir: {}\troi_features_dir: {}".format(roi_boxes_dir,roi_features_dir))
        self.roi_boxes_dir=roi_boxes_dir
        self.roi_features_dir=roi_features_dir
        self.max_num_rois=max_num_rois

        self.bert_model_dir=bert_model_dir
        self.__create_classifier_model()

        self.device=torch.device("cpu") #デフォルトではCPU

    def __create_classifier_model(self):
        logger=self.logger

        self.classifier_model=None
        if self.bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERTモデルを用いて分類器のパラメータを初期化します。")
            config=BertConfig.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
            self.classifier_model=ImageBertForMultipleChoice(config)
            self.classifier_model.initialize_from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込んで分類器のパラメータを初期化します。".format(self.bert_model_dir))
            config=BertConfig.from_json_file(self.bert_model_dir)
            self.classifier_model=ImageBertForMultipleChoice(config)
            self.classifier_model.initialize_from_pretrained(self.bert_model_dir)

    def to(self,device:torch.device):
        self.device=device
        self.classifier_model.to(device)

    def test(
        self,
        model_filepath:str,
        result_filepath:str,
        labels_filepath:str):
        logger=self.logger
        logger.info("モデルのテストを開始します。")

        #モデルのパラメータを読み込む。
        logger.info("{}からモデルパラメータを読み込みます。".format(model_filepath))
        parameters=torch.load(model_filepath,map_location=self.device)
        self.classifier_model.load_state_dict(parameters)

        #評価
        res=evaluate(
            self.classifier_model,
            self.test_options,
            self.roi_boxes_dir,
            self.roi_features_dir,
            self.test_dataloader,
            max_num_rois=self.max_num_rois,
            device=self.device
        )
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
