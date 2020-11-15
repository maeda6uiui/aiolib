"""
問題文とコンテキストをBERTモデルに入力するためにエンコードする。
"""
import argparse
import gzip
import json
import logging
import os
import torch
from tqdm import tqdm
from transformers import BertJapaneseTokenizer
from typing import Dict,List

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

BERT_MAX_SEQ_LENGTH=512

class InputExample(object):
    """
    問題
    """
    def __init__(self, qid:str, question:str, options:List[str], label:int):
        self.qid = qid
        self.question = question
        self.options = options
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
    contexts:Dict[str,str])->Dict[str,torch.Tensor]:
    """
    問題をエンコードする。
    """
    #最初の問題の選択肢の数を代表値として取得する。
    num_options=len(examples[0].options)

    input_ids=torch.empty(len(examples),num_options,BERT_MAX_SEQ_LENGTH,dtype=torch.long)
    attention_mask=torch.empty(len(examples),num_options,BERT_MAX_SEQ_LENGTH,dtype=torch.long)
    token_type_ids=torch.empty(len(examples),num_options,BERT_MAX_SEQ_LENGTH,dtype=torch.long)
    labels=torch.empty(len(examples),dtype=torch.long)

    for example_index,example in enumerate(tqdm(examples)):
        for option_index,option in enumerate(example.options):
            text_a=example.question+tokenizer.sep_token+option
            text_b=contexts[option]

            encoding = tokenizer.encode_plus(
                text_a,
                text_b,
                return_tensors="pt",
                add_special_tokens=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                max_length=BERT_MAX_SEQ_LENGTH,
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

def main(args):
    example_filepath:str=args.example_filepath
    context_filepath:str=args.context_filepath
    bert_model_dir:str=args.bert_model_dir
    save_dir:str=args.save_dir

    logger.info("{}から問題を読み込みます。".format(example_filepath))
    examples=load_examples(example_filepath)

    logger.info("{}からコンテキストを読み込みます。".format(context_filepath))
    contexts=load_contexts(context_filepath)

    logger.info("{}から語彙を読み込みます。".format(bert_model_dir))
    tokenizer=BertJapaneseTokenizer.from_pretrained(bert_model_dir)

    logger.info("問題のエンコードを開始します。")
    encoding=encode_examples(tokenizer,examples,contexts)

    os.makedirs(save_dir,exist_ok=True)
    input_ids_filepath=os.path.join(save_dir,"input_ids.pt")
    token_type_ids_filepath=os.path.join(save_dir,"token_type_ids.pt")
    attention_mask_filepath=os.path.join(save_dir,"attention_mask.pt")
    labels_filepath=os.path.join(save_dir,"labels.pt")

    torch.save(encoding["input_ids"],input_ids_filepath)
    torch.save(encoding["token_type_ids"],token_type_ids_filepath)
    torch.save(encoding["attention_mask"],attention_mask_filepath)
    torch.save(encoding["labels"],labels_filepath)

    logger.info("処理を完了しました。")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--example_filepath",type=str)
    parser.add_argument("--context_filepath",type=str)
    parser.add_argument("--bert_model_dir",type=str)
    parser.add_argument("--save_dir",type=str)
    args=parser.parse_args()

    main(args)
