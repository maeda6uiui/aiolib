"""
BM25を適用するために必要なデータの準備を行う。
"""
import collections
import glob
import gzip
import json
import logging
import os
import MeCab
from tqdm import tqdm
from typing import Dict,List,Tuple

from .. import hashing

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

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

def count_genkeis(mecab:MeCab,context:str)->(collections.Counter,int):
    """
    コンテキストに対して形態素解析を行い、出現する単語(原形)をカウントする。
    """
    genkeis=[]

    node=mecab.parseToNode(context)
    while node:
        features=node.feature.split(",")

        hinsi=features[0]
        if hinsi=="BOS/EOS":
            node=node.next
            continue

        genkei=features[6]
        genkeis.append(genkei)

        node=node.next

    counter=collections.Counter(genkeis)
    return counter,len(genkeis)

class BM25Preprocessing(object):
    """
    BM25を適用するために必要な前処理を行う。
    """
    def __init__(self,logger:logging.Logger=default_logger):
        self.logger=logger

    def __count_genkeis(self,context_filepath:str,count_save_dir:str):
        logger=self.logger
        mecab=MeCab.Tagger()

        logger.info("{}からコンテキストを読み込みます。".format(context_filepath))
        contexts=load_contexts(context_filepath)

        #各コンテキストに対して形態素解析を行い単語をカウントする。
        logger.info("各コンテキストに対して形態素解析を行い単語をカウントします。")
        os.makedirs(count_save_dir,exist_ok=True)

        total_num_words=0
        for title,context in tqdm(contexts.items()):
            counter,num_words=count_genkeis(mecab,context)
            total_num_words+=num_words

            #カウントの結果をテキストファイルに出力する。
            title_hash=hashing.get_md5_hash(title)
            count_filepath=os.path.join(count_save_dir,title_hash+".txt")

            with open(count_filepath,"w",encoding="utf_8",newline="") as w:
                w.write(str(num_words)+"\n")    #1行目はその文書に含まれる単語数

                #2行目からは各単語の出現頻度 (単語,出現頻度)
                for genkei,freq in counter.most_common():
                    w.write(genkei)
                    w.write("\t")
                    w.write(str(freq))
                    w.write("\n")

        avgdl=total_num_words/len(contexts)
        logger.info("avgdl: {}".format(avgdl))

    def __count_nqis(self,count_save_dir:str,save_dir:str)->List[Tuple[str,int]]:
        logger=self.logger

        #ある単語が含まれる文書の数をカウントする。
        logger.info("単語が含まれる文書の数をカウントします。")

        genkeis=[]
        pathname=os.path.join(count_save_dir,"*.txt")
        count_files=glob.glob(pathname)
        for count_file in tqdm(count_files):
            with open(count_file,"r",encoding="utf_8") as r:
                lines=r.read().splitlines()

            for i in range(1,len(lines)):
                genkei=lines[i].split("\t")[0]
                genkeis.append(genkei)

        nqis_counter=collections.Counter(genkeis)

        nqis_filepath=os.path.join(save_dir,"nqis.txt")
        nqis_most_common=nqis_counter.most_common()
        with open(nqis_filepath,"w",encoding="utf_8",newline="") as w:
            for genkei,freq in nqis_most_common:
                w.write(genkei)
                w.write("\t")
                w.write(str(freq))
                w.write("\n")

        return nqis_most_common

    def preprocess(self,context_filepath:str,save_dir:str,ignore_tok_k:int=30):
        os.makedirs(save_dir,exist_ok=True)
        count_save_dir=os.path.join(save_dir,"Count")
        self.__count_genkeis(context_filepath,count_save_dir)
        nqis_most_common=self.__count_nqis(count_save_dir,save_dir)

        #上位k位の頻出単語はignores.txtに保存しておく。
        ignores_filepath=os.path.join(save_dir,"ignores.txt")
        with open(ignores_filepath,"w",encoding="utf_8",newline="") as w:
            for idx,(genkei,freq) in enumerate(nqis_most_common):
                if idx>=ignore_tok_k:
                    break

                w.write(genkei+"\n")