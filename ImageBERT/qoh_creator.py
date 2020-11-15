"""
問題のインデックス(通し番号)と各選択肢のハッシュ値を計算し、
その結果をJSONファイルに保存する。
"""
import argparse
import hashlib
import json
import logging
from typing import List

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def get_md5_hash(v:str)->str:
    return hashlib.md5(v.encode()).hexdigest()

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

def create_qohs_json(examples:List[InputExample],save_filepath:str):
    lines=[]
    for idx,example in enumerate(examples):
        options=example.options
        option_hashes=[]
        for option in options:
            option_hash=get_md5_hash(option)
            option_hashes.append(option_hash)

        qoh={
            "question_index":idx,
            "option_hashes":option_hashes
        }

        line=json.dumps(qoh,ensure_ascii=False)
        lines.append(line)

    with open(save_filepath,"w",encoding="utf_8") as w:
        for line in lines:
            w.write(line+"\n")
    
def main(example_filepath:str,save_filepath:str):
    logger.info("問題を読み込みます。")
    examples=load_examples(example_filepath)
    logger.info("問題情報を作成します。")
    create_qohs_json(examples,save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--example_filepath",type=str)
    parser.add_argument("--save_filepath",type=str)
    args=parser.parse_args()

    main(
        args.example_filepath,
        args.save_filepath
    )
