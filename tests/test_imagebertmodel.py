import logging
import sys
import os
import torch
sys.path.append(os.path.abspath("../src/aiolib"))

from transformers import(
    BertConfig,
    BertJapaneseTokenizer,
    BertModel
)

from modeling.imagebert.models import ImageBertModel

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    pretrained_model_name="cl-tohoku/bert-base-japanese-whole-word-masking"

    #ImageBERTモデルの作成
    config=BertConfig.from_pretrained(pretrained_model_name)
    im_bert=ImageBertModel.from_pretrained(pretrained_model_name)
    im_bert.to(device)
    im_bert.eval()

    #BERTモデルの作成
    bert=BertModel.from_pretrained(pretrained_model_name)
    bert.to(device)
    bert.eval()

    #入力するテキストのEncode
    tokenizer=BertJapaneseTokenizer.from_pretrained(pretrained_model_name)

    text=(
        "アメリカ合衆国（アメリカがっしゅうこく、英: United States of America, USA、"
        "通称アメリカは、北アメリカ、太平洋およびカリブに位置する連邦共和制国家。"
        "首都はコロンビア特別区（ワシントンD.C.）。"
    )
    encoding=tokenizer.encode_plus(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        max_length=512,
        truncation=True
    )

    input_ids=encoding["input_ids"]
    attention_mask=encoding["attention_mask"]
    token_type_ids=encoding["token_type_ids"]

    #RoI特徴量の読み込み
    roi_boxes=torch.load("./Data/roi_boxes.pt",map_location=device)
    roi_features=torch.load("./Data/roi_features.pt",map_location=device)

    roi_boxes=roi_boxes[:100]
    roi_features=roi_features[:100]

    roi_boxes=roi_boxes.unsqueeze(0)
    roi_features=roi_features.unsqueeze(0)

    """
    #入力データの作成
    im_bert_inputs={
        "input_ids":input_ids,
        "roi_boxes":roi_boxes,
        "roi_features":roi_features,
        "max_num_rois":100
    }
    bert_inputs={
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "token_type_ids":token_type_ids
    }

    with torch.no_grad():
        im_bert_outputs=im_bert(**im_bert_inputs)
        bert_outputs=bert(**bert_inputs)

    im_bert_logits=im_bert_outputs[0]
    bert_logits=bert_outputs[0]

    print("im_bert_logits:")
    print(im_bert_logits)

    print("bert_logits:")
    print(bert_logits)
    """

    word_embedding_1=bert.get_input_embeddings()
    word_embedding_2=im_bert.get_input_embeddings()
    embeddings_1=word_embedding_1(input_ids)
    embeddings_2=word_embedding_2(input_ids)

    print("embeddings_1:")
    print(embeddings_1)
    print("embeddings_2:")
    print(embeddings_2)

if __name__=="__main__":
    main()
