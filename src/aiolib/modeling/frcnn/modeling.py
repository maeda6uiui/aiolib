import logging
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from transformers import (
    BertConfig,
    BertModel,
    BertForMultipleChoice,
    AdamW,
    get_linear_schedule_with_warmup,
)
from typing import List,Tuple

sys.path.append("../../")
from util import hashing

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

class Options(object):
    """
    問題の選択肢
    """
    def __init__(self):
        self.options=[]

    def append(self,option:str):
        self.options.append(option)

    def get(self,index:int):
        return self.options[index]

def load_options_list(list_filepath:str,logger:logging.Logger=default_logger)->List[Options]:
    logger.info("{}から選択肢のリストを読み込みます。".format(list_filepath))

    with open(list_filepath,"r",encoding="UTF-8") as r:
        lines=r.read().splitlines()

    options=[]
    ops=None
    for line in lines:
        if ops is None:
            ops=Options()

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

def create_text_embeddings(
    bert_model:BertModel,
    input_ids:torch.Tensor,   #(num_options,max_seq_length,embedding_dim)
    max_seq_length:int=512,
    embedding_dim:int=768)->torch.Tensor:
    """
    テキストEmbeddingを作成する。
    返されるTensorのサイズは(num_options,max_seq_length,embedding_dim)
    """
    bert_model.eval()
    num_options=input_ids.size(0)
    ret=torch.empty(num_options,max_seq_length,embedding_dim).to(device)

    attention_mask=torch.ones(num_options,max_seq_length,dtype=torch.long).to(device)
    #テキストのToken Type IDは0
    token_type_ids=torch.zeros(num_options,max_seq_length,dtype=torch.long).to(device)

    for i in range(num_options):
        op_input_ids=input_ids[i].unsqueeze(0).to(device)
        op_attention_mask=attention_mask[i].unsqueeze(0).to(device)
        op_token_type_ids=token_type_ids[i].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs=bert_model(
                input_ids=op_input_ids,
                attention_mask=op_attention_mask,
                token_type_ids=op_token_type_ids
            )
            embeddings=bert_model.get_input_embeddings()

            ret[i]=embeddings(op_input_ids)

    return ret

def process_roi_embeddings(
    bert_model:BertModel,
    roi_boxes_embeddings:torch.Tensor,
    roi_features_embeddings:torch.Tensor,
    max_seq_length:int=512,
    embedding_dim:int=768,
    max_num_rois:int=100)->Tuple[torch.Tensor,int]:
    """
    RoIのEmbeddingを作成する。
    """
    bert_model.eval()

    position_embeddings=bert_model.embeddings.position_embeddings
    token_type_embeddings=bert_model.embeddings.token_type_embeddings
    layer_norm=bert_model.embeddings.layer_norm
    dropout=bert_model.embeddings.dropout

    num_rois=roi_boxes_embeddings.size(0)
    if num_rois>max_num_rois:
        roi_boxes_embeddings=roi_boxes_embeddings[:max_num_rois]
        roi_features_embeddings=roi_features_embeddings[:max_num_rois]
        num_rois=max_num_rois

    position_ids=torch.empty(max_seq_length,dtype=torch.long).to(device)
    for i in range(max_seq_length):
        position_ids[i]=i
    position_ids=position_ids[max_seq_length-num_rois:]

    #RoIのToken Type IDは1
    token_type_ids=torch.ones(max_seq_length,dtype=torch.long).to(device)
    token_type_ids=token_type_ids[max_seq_length-num_rois:]

    v_position_embeddings=position_embeddings(position_ids)
    v_token_type_embeddings=token_type_embeddings(token_type_ids)

    embeddings=roi_boxes_embeddings+roi_features_embeddings+v_position_embeddings+v_token_type_embeddings
    embeddings=layer_norm(embeddings)
    embeddings=dropout(embeddings)

    return embeddings,num_rois

def create_inputs_embeds(
    bert_model:BertModel,
    input_ids:torch.Tensor,
    indices:torch.Tensor,
    options:List[Options],
    im_boxes_dir:str,
    im_features_dir:str,
    max_seq_length:int=512,
    embedding_dim:int=768,
    max_num_rois:int=100)->torch.Tensor:
    """
    BertForMultipleChoiceに入力するEmbeddingを作成する。
    """
    batch_size=input_ids.size(0)
    num_options=input_ids.size(1)

    inputs_embeds=torch.empty(batch_size,num_options,max_seq_length,embedding_dim).to(device)

    for i in range(batch_size):
        text_embeddings=create_text_embeddings(
            bert_model,input_ids[i],
            max_seq_length=max_seq_length,
            embedding_dim=embedding_dim
        )

        ops=options[indices[i]]
        for j in range(num_options):
            article_name=ops.get(j)
            title_hash=hashing.get_md5_hash(article_name)

            option_embeddings=None
            im_boxes_filepath=os.path.join(im_boxes_dir,title_hash+".pt")
            im_features_filepath=os.path.join(im_features_dir,title_hash+".pt")

            #画像の特徴量が存在する場合 (矩形領域の座標データも存在するはず)
            if os.path.exists(im_features_filepath):
                roi_boxes=torch.load(im_boxes_filepath,map_location=device).to(device)
                roi_features=torch.load(im_features_filepath,map_location=device).to(device)

                roi_embeddings,num_rois=process_roi_embeddings(
                    bert_model,
                    roi_boxes,
                    roi_features,
                    max_seq_length=max_seq_length,
                    embedding_dim=embedding_dim,
                    max_num_rois=max_num_rois
                )

                trunc_text_embeddings=text_embeddings[j,:max_seq_length-num_rois]
                option_embeddings=torch.cat([trunc_text_embeddings,roi_embeddings],dim=0)
            #画像の特徴量が存在しない場合
            else:
                option_embeddings=text_embeddings[j]

            inputs_embeds[i,j]=option_embeddings

    return inputs_embeds

def train(
    bert_model:BertModel,
    classifier_model:BertForMultipleChoice,
    options:List[Options],
    im_boxes_dir:str,
    im_features_dir:str,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LambdaLR,
    dataloader:DataLoader,
    max_seq_length:int=512,
    embedding_dim:int=768,
    max_num_rois:int=100,
    logger:logging.Logger=default_logger,
    logging_steps:int=100)->float:
    """
    モデルの訓練を行う。
    """
    bert_model.eval()
    classifier_model.train()

    count_steps=0
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
        batch = tuple(t for t in batch)

        bert_inputs = {
            "indices":batch[0].to(device),
            "input_ids": batch[1].to(device),
            #"attention_mask": batch[2].to(device),
            #"token_type_ids": batch[3].to(device),
            "labels": batch[4].to(device)
        }

        inputs_embeds=create_inputs_embeds(
            bert_model,
            bert_inputs["input_ids"],
            bert_inputs["indices"],
            options,
            im_boxes_dir,
            im_features_dir,
            max_seq_length=max_seq_length,
            embedding_dim=embedding_dim,
            max_num_rois=max_num_rois
        )

        classifier_inputs={
            "inputs_embeds":inputs_embeds,
            #"attention_mask":bert_inputs["attention_mask"],
            #"token_type_ids":inputs_token_type_ids,
            "labels":bert_inputs["labels"]
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
    bert_model:BertModel,
    classifier_model:BertForMultipleChoice,
    options:List[Options],
    im_boxes_dir:str,
    im_features_dir:str,
    dataloader:DataLoader,
    max_seq_length:int=512,
    embedding_dim:int=768,
    max_num_rois:int=100):
    """
    モデルの評価を行う。
    結果やラベルはDict形式で返される。
    """
    bert_model.eval()
    classifier_model.eval()

    count_steps=0
    total_loss=0

    preds = None
    correct_labels = None

    for batch_idx,batch in enumerate(dataloader):
        with torch.no_grad():
            batch_size=len(batch)
            batch = tuple(t for t in batch)

            bert_inputs = {
                "indices":batch[0].to(device),
                "input_ids": batch[1].to(device),
                "attention_mask": batch[2].to(device),
                "token_type_ids": batch[3].to(device),
                "labels": batch[4].to(device)
            }

            inputs_embeds=create_inputs_embeds(
                bert_model,
                bert_inputs["input_ids"],
                bert_inputs["indices"],
                options,
                im_boxes_dir,
                im_features_dir,
                max_seq_length=max_seq_length,
                embedding_dim=embedding_dim,
                max_num_rois=max_num_rois
            )

            classifier_inputs={
                "inputs_embeds":inputs_embeds,
                #"attention_mask":bert_inputs["attention_mask"],
                #"token_type_ids":inputs_token_type_ids,
                "labels":bert_inputs["labels"]
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

class FasterRCNNModeler(object):
    """
    Faster R-CNNで抽出された画像特徴量を使用してモデルを訓練する。
    """
    def __init__(
        self,
        train_input_dir:str,
        dev_input_dir:str,
        bert_model_dir:str,
        im_boxes_dir:str,
        im_features_dir:str,
        seed:int=42,
        logger:logging.Logger=default_logger):
        self.logger=logger

        logger.info("シード: {}".format(seed))
        set_seed(seed)

        logger.info("{}から訓練用データセットを作成します。".format(train_input_dir))
        self.train_dataset=create_dataset(train_input_dir,num_examples=-1,num_options=4)

        logger.info("{}からDev用データローダを作成します。".format(dev_input_dir))
        dev_dataset=create_dataset(dev_input_dir,num_examples=-1,num_options=20)
        self.dev_dataloader=DataLoader(dev_dataset,batch_size=4,shuffle=False)

        logger.info("選択肢のリストを読み込みます。")
        self.train_options=load_options_list(os.path.join(train_input_dir,"options_list.txt"))
        self.dev_options=load_options_list(os.path.join(dev_input_dir,"options_list.txt"))

        logger.info("im_boxes_dir: {}\tim_features_dir: {}".format(im_boxes_dir,im_features_dir))
        self.im_boxes_dir=im_boxes_dir
        self.im_features_dir=im_features_dir

        self.bert_model_dir=bert_model_dir
        self.__create_bert_model(bert_model_dir)
        self.__create_classifier_model(bert_model_dir)

    def __create_bert_model(self,bert_model_dir:str):
        logger=self.logger

        self.bert_model=None
        if bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERTモデルを読み込みます。")
            self.embedding_dim=768
            self.bert_model=BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込みます。".format(bert_model_dir))
            config=BertConfig.from_json_file(os.path.join(bert_model_dir,"bert_config.json"))
            self.embedding_dim=config["hidden_size"]
            self.bert_model=BertModel.from_pretrained(bert_model_dir,config=config)
        self.bert_model.to(device)

    def __create_classifier_model(self,bert_model_dir:str):
        logger=self.logger

        self.classifier_model=None
        if bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERTモデルを用いて分類器のパラメータを初期化します。")
            self.classifier_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込んで分類器のパラメータを初期化します。".format(bert_model_dir))
            self.classifier_model=BertForMultipleChoice.from_pretrained(bert_model_dir)
        self.classifier_model.to(device)

    def train_and_eval(
        self,
        train_batch_size:int=4,
        num_epochs:int=5,
        lr:float=2.5e-5,
        init_parameters=False,
        max_num_rois:int=100,
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
                self.bert_model,
                self.classifier_model,
                self.train_options,
                self.im_boxes_dir,
                self.im_features_dir,
                optimizer,
                scheduler,
                train_dataloader,
                max_seq_length=512,
                embedding_dim=self.embedding_dim,
                max_num_rois=max_num_rois,
                logger=logger,
                logging_steps=logging_steps
            )
            logger.info("訓練時の損失平均値: {}".format(mean_loss))

            #チェックポイントを保存する。
            checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
            torch.save(self.classifier_model.state_dict(),checkpoint_filepath)

            #評価
            res=evaluate(
                self.bert_model,
                self.classifier_model,
                self.dev_options,
                self.im_boxes_dir,
                self.im_features_dir,
                self.dev_dataloader,
                max_seq_length=512,
                embedding_dim=self.embedding_dim,
                max_num_rois=max_num_rois
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

class FasterRCNNTester(object):
    """
    Faster R-CNNで抽出された特徴量を使用して訓練されたモデルをテストする。
    """
    def __init__(
        self,
        test_input_dir:str,
        bert_model_dir:str,
        im_boxes_dir:str,
        im_features_dir:str,
        seed:int=42,
        logger:logging.Logger=default_logger):
        self.logger=logger

        logger.info("シード: {}".format(seed))
        set_seed(seed)

        logger.info("{}からテスト用データローダを作成します。".format(test_input_dir))
        test_dataset=create_dataset(test_input_dir,num_examples=-1,num_options=20)
        self.test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=False)

        logger.info("選択肢のリストを読み込みます。")
        self.test_options=load_options_list(os.path.join(test_input_dir,"options_list.txt"))

        self.__create_bert_model(bert_model_dir)
        self.__create_classifier_model(bert_model_dir)

        logger.info("im_boxes_dir: {}\tim_features_dir: {}".format(im_boxes_dir,im_features_dir))
        self.im_boxes_dir=im_boxes_dir
        self.im_features_dir=im_features_dir

    def __create_bert_model(self,bert_model_dir:str):
        logger=self.logger

        self.bert_model=None
        if bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERTモデルを読み込みます。")
            self.embedding_dim=768
            self.bert_model=BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込みます。".format(bert_model_dir))
            config=BertConfig.from_json_file(os.path.join(bert_model_dir,"bert_config.json"))
            self.embedding_dim=config["hidden_size"]
            self.bert_model=BertModel.from_pretrained(bert_model_dir,config=config)
        self.bert_model.to(device)

    def __create_classifier_model(self,bert_model_dir:str):
        logger=self.logger

        self.classifier_model=None
        if bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトのBERTモデルを用いて分類器のパラメータを初期化します。")
            self.classifier_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}からBERTモデルを読み込んで分類器のパラメータを初期化します。".format(bert_model_dir))
            self.classifier_model=BertForMultipleChoice.from_pretrained(bert_model_dir)
        self.classifier_model.to(device)

    def test(
        self,
        model_filepath:str,
        result_filepath:str,
        labels_filepath:str,
        max_num_rois:int=100):
        logger=self.logger
        logger.info("モデルのテストを開始します。")

        #モデルのパラメータを読み込む。
        logger.info("{}からモデルパラメータを読み込みます。".format(model_filepath))
        parameters=torch.load(model_filepath,map_location=device)
        self.classifier_model.load_state_dict(parameters)

        #評価
        res=evaluate(
            self.bert_model,
            self.classifier_model,
            self.dev_options,
            self.im_boxes_dir,
            self.im_features_dir,
            self.dev_dataloader,
            max_seq_length=512,
            embedding_dim=self.embedding_dim,
            max_num_rois=max_num_rois
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
