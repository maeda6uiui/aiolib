"""
問題文とコンテキストをBERTモデルに入力するためにエンコードする。
"""
import gzip
import json
import logging
import os
import torch
from tqdm import tqdm
from transformers import BertJapaneseTokenizer
from typing import Dict,List

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

class InputExample(object):
    """
    問題
    """
    def __init__(self, qid:str, question:str, endings:List[str], label:int):
        self.qid = qid
        self.question = question
        self.endings = endings
        self.label = label

def load_examples(example_filepath:str)->List[InputExample]:
    """
    問題を読み込む。
    """
    with open(example_filepath, "r", encoding="utf_8") as r:
        lines = r.read().splitlines()

    examples = []
    for line in lines:
        data = json.loads(line)

        qid = data["qid"]
        question = data["question"].replace("_", "")
        options = data["answer_candidates"]
        answer = data["answer_entity"]

        label=0
        if answer!="":
            label=options.index(answer)

        example = InputExample(qid, question, options, label)
        examples.append(example)

    return examples

def load_contexts(context_filepath:str)->Dict[str,str]:
    """
    コンテキスト(Wikipedia記事)を読み込む。
    """
    contexts={}

    with gzip.open(context_filepath,mode="rt",encoding="utf-8") as r:
        for line in r:
            data = json.loads(line)

            title=data["title"]
            text=data["text"]

            contexts[title]=text

    return contexts

def encode_examples(
    tokenizer:BertJapaneseTokenizer,
    examples:List[InputExample],
    contexts:Dict[str,str],
    max_seq_length:int,
    logger:logging.Logger)->Dict[str,torch.Tensor]:
    #最初の問題の選択肢の数を代表値として取得する。
    num_options=len(examples[0].endings)

    input_ids=torch.empty(len(examples),num_options,max_seq_length,dtype=torch.long)
    attention_mask=torch.empty(len(examples),num_options,max_seq_length,dtype=torch.long)
    token_type_ids=torch.empty(len(examples),num_options,max_seq_length,dtype=torch.long)
    labels=torch.empty(len(examples),dtype=torch.long)

    for example_index,example in enumerate(tqdm(examples)):
        for option_index,ending in enumerate(example.endings):
            text_a=example.question+tokenizer.sep_token+ending
            text_b=contexts[ending]

            encoding = tokenizer.encode_plus(
                text_a,
                text_b,
                return_tensors="pt",
                add_special_tokens=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                max_length=max_seq_length,
                truncation=True,
                truncation_strategy="only_second"   #コンテキストをtruncateする。
            )

            input_ids_tmp=encoding["input_ids"].view(-1)
            token_type_ids_tmp=encoding["token_type_ids"].view(-1)
            attention_mask_tmp=encoding["attention_mask"].view(-1)

            input_ids[example_index,option_index]=input_ids_tmp
            token_type_ids[example_index,option_index]=token_type_ids_tmp
            attention_mask[example_index,option_index]=attention_mask_tmp

            if example_index==0 and option_index<4:
                logger.info("option_index={}".format(option_index))
                logger.info("text_a: {}".format(text_a[:512]))
                logger.info("text_b: {}".format(text_b[:512]))
                logger.info("input_ids: {}".format(input_ids_tmp.detach().cpu().numpy()))
                logger.info("token_type_ids: {}".format(token_type_ids_tmp.detach().cpu().numpy()))
                logger.info("attention_mask: {}".format(attention_mask_tmp.detach().cpu().numpy()))

        labels[example_index]=example.label

    ret={
        "input_ids":input_ids,
        "token_type_ids":token_type_ids,
        "attention_mask":attention_mask,
        "labels":labels
    }

    return ret

class ExampleEncoder(object):
    """
    問題とコンテキスト(Wikipedia記事)を読み込んでエンコードする。
    """
    def __init__(
        self,
        context_filepath:str,
        bert_model_dir:str,
        max_seq_length:int=512,
        logger:logging.Logger=default_logger):
        logger.info("{}からコンテキストを読み込みます。".format(context_filepath))
        self.contexts=load_contexts(context_filepath)

        self.tokenizer=None
        if bert_model_dir=="USE_DEFAULT":
            logger.info("デフォルトの語彙を使用します。")
            self.tokenizer=BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        else:
            logger.info("{}から語彙を読み込みます。".format(bert_model_dir))
            self.tokenizer=BertJapaneseTokenizer.from_pretrained(bert_model_dir)

        self.max_seq_length=max_seq_length
        self.logger=logger

    def encode(self,example_filepath:str)->Dict[str,torch.Tensor]:
        logger.info("{}から問題を読み込みます。".format(example_filepath))
        examples=load_examples(example_filepath)

        ret=encode_examples(self.tokenizer,examples,self.contexts,self.max_seq_length,self.logger)
        return ret

    def encode_save(self,example_filepath:str,save_dir:str):
        encoded=self.encode(example_filepath)
        
        os.makedirs(save_dir,exist_ok=True)
        input_ids_filepath=os.path.join(save_dir,"input_ids.pt")
        token_type_ids_filepath=os.path.join(save_dir,"token_type_ids.pt")
        attention_mask_filepath=os.path.join(save_dir,"attention_mask.pt")
        labels_filepath=os.path.join(save_dir,"labels.pt")

        torch.save(encoded["input_ids"],input_ids_filepath)
        torch.save(encoded["token_type_ids"],token_type_ids_filepath)
        torch.save(encoded["attention_mask"],attention_mask_filepath)
        torch.save(encoded["labels"],labels_filepath)    
