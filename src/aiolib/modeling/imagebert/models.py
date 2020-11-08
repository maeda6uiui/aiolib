import logging
import torch
import torch.nn as nn
from transformers import(
    BertConfig,
    BertJapaneseTokenizer,
    BertModel,
    BertPreTrainedModel
)

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

BERT_MAX_SEQ_LENGTH=512 #BERTに入力するシーケンスの最大長

class ImageBertModel(BertModel):
    """
    ImageBERTのモデル
    """
    def __init__(
        self,
        config:BertConfig,
        add_pooling_layer:bool=True,
        roi_features_dim:int=1024,  #RoI特徴量の次元
        image_width:int=256,    #元画像の幅
        image_height:int=256,   #元画像の高さ
        logger:logging.Logger=default_logger):
        super().__init__(config,add_pooling_layer=add_pooling_layer)
        self.logger=logger
        self.roi_features_dim=roi_features_dim

        #FC層の作成
        #RoI関連のベクトルをBERTのhidden sizeに射影する。
        self.fc_roi_boxes=nn.Linear(5,config.hidden_size)
        self.fc_roi_features=nn.Linear(roi_features_dim,config.hidden_size)

        self.init_weights()

        #Position ID (トークンのインデックス)
        self.position_ids=torch.empty(BERT_MAX_SEQ_LENGTH,dtype=torch.long)
        for i in range(BERT_MAX_SEQ_LENGTH):
            self.position_ids[i]=i
        #テキストのToken Type IDは0
        self.text_token_type_ids=torch.zeros(BERT_MAX_SEQ_LENGTH,dtype=torch.long)
        #RoIのToken Type IDは1
        self.roi_token_type_ids=torch.ones(BERT_MAX_SEQ_LENGTH,dtype=torch.long)

        self.wh_tensor=torch.empty(5)  #(RoIの)Position Embedding作成に使用する。
        self.wh_tensor[0]=image_width
        self.wh_tensor[1]=image_height
        self.wh_tensor[2]=image_width
        self.wh_tensor[3]=image_height
        self.wh_tensor[4]=image_width*image_height

        self.sep_embedding=None

    def create_sep_embedding(self,pretrained_model_name_or_path:str):
        """
        [SEP]トークンのEmbeddingを作成する。
        """
        logger=self.logger

        logger.info("[SEP]トークンのEmbeddingを作成します。")

        tokenizer=BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path)
        bert_model=BertModel.from_pretrained(pretrained_model_name_or_path)

        input_ids=torch.tensor([tokenizer.sep_token_id])

        word_embeddings=bert_model.get_input_embeddings()
        self.sep_embedding=word_embeddings(input_ids)
        self.sep_embedding=torch.squeeze(self.sep_embedding)

    def to(self,device:torch.device):
        super().to(device)

        self.fc_roi_boxes.to(device)
        self.fc_roi_features.to(device)
        self.position_ids=self.position_ids.to(device)
        self.text_token_type_ids=self.text_token_type_ids.to(device)
        self.roi_token_type_ids=self.roi_token_type_ids.to(device)
        self.wh_tensor=self.wh_tensor.to(device)

    def __create_embeddings(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor, #(N,max_num_rois,4)
        roi_features:torch.Tensor,   #(N,max_num_rois,roi_features_dim)
        max_num_rois:int,
    )->torch.Tensor:
        """
        入力Embeddingを作成する。
        """
        word_embeddings=self.embeddings.word_embeddings
        position_embeddings=self.embeddings.position_embeddings
        token_type_ids_embeddings=self.embeddings.token_type_embeddings
        layer_norm=self.embeddings.LayerNorm
        dropout=self.embeddings.dropout

        batch_size=input_ids.size(0)

        v_position_embeddings=position_embeddings(self.position_ids)
        v_text_token_type_ids_embeddings=token_type_ids_embeddings(self.text_token_type_ids)
        v_roi_token_type_ids_embeddings=token_type_ids_embeddings(self.roi_token_type_ids)

        #=== テキストEmbeddingを作成する。===
        v_word_embeddings=word_embeddings(input_ids)    #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        #=== RoIのEmbeddingを作成する。 ===
        roi_features_embeddings=self.fc_roi_features(roi_features)
        #(N,max_num_rois,hidden_size)

        #RoIの座標から(RoIの)Position Embeddingを作成する。
        roi_position_embeddings=torch.empty(batch_size,max_num_rois,5).to(self.device)
        for i in range(batch_size):
            for j in range(max_num_rois):
                x_tl=roi_boxes[i,j,0]
                y_tl=roi_boxes[i,j,1]
                x_br=roi_boxes[i,j,2]
                y_br=roi_boxes[i,j,3]

                roi_position_embeddings[i,j,0]=x_tl
                roi_position_embeddings[i,j,1]=y_tl
                roi_position_embeddings[i,j,2]=x_br
                roi_position_embeddings[i,j,3]=y_br
                roi_position_embeddings[i,j,4]=(x_br-x_tl)*(y_br-y_tl)

        roi_position_embeddings=torch.div(roi_position_embeddings,self.wh_tensor)

        #RoIのPosition Embeddingを射影する。
        roi_position_embeddings=self.fc_roi_boxes(roi_position_embeddings)
        #(N,max_num_rois,hidden_size)

        roi_embeddings=roi_features_embeddings+roi_position_embeddings

        #=== テキストEmbeddingとRoI Embeddingを結合する。
        trunc_word_embeddings=v_word_embeddings[:,:BERT_MAX_SEQ_LENGTH-max_num_rois,:]

        text_roi_embeddings=torch.cat([trunc_word_embeddings,roi_embeddings],dim=1)
        #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        #[SEP]トークンのEmbeddingを入れる。
        if self.sep_embedding is not None:
            text_roi_embeddings[:,BERT_MAX_SEQ_LENGTH-max_num_rois-1]=self.sep_embedding.detach()
            text_roi_embeddings[:,-1]=self.sep_embedding.detach()

        trunc_text_token_type_ids_embeddings=v_text_token_type_ids_embeddings[:BERT_MAX_SEQ_LENGTH-max_num_rois]
        trunc_roi_token_type_ids_embeddings=v_roi_token_type_ids_embeddings[BERT_MAX_SEQ_LENGTH-max_num_rois:]
        v_token_type_ids_embeddings=torch.cat([trunc_text_token_type_ids_embeddings,trunc_roi_token_type_ids_embeddings],dim=0)
        #(BERT_MAX_SEQ_LENGTH,hidden_size)

        #Position EmbeddingとToken Type ID EmbeddingをExpandする。
        v_position_embeddings=v_position_embeddings.expand(batch_size,BERT_MAX_SEQ_LENGTH,-1)
        #(N,BERT_MAX_SEQ_LENGTH,hidden_size)
        v_token_type_ids_embeddings=v_token_type_ids_embeddings.expand(batch_size,BERT_MAX_SEQ_LENGTH,-1)
        #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        #最終的なEmbeddingはすべてを足したもの
        embeddings=text_roi_embeddings+v_position_embeddings+v_token_type_ids_embeddings
        embeddings=layer_norm(embeddings)
        embeddings=dropout(embeddings)  #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        return embeddings

    def forward(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor,    #(N,max_num_rois,4)
        roi_features:torch.Tensor,  #(N,max_num_rois,roi_features_dim)
        max_num_rois:int,   #入力するRoIの最大数
        output_hidden_states:bool=None,
        return_dict:bool=None):
        embeddings=self.__create_embeddings(input_ids,roi_boxes,roi_features,max_num_rois)

        #Todo: とりあえずattention_maskはすべて1
        attention_mask=torch.ones(input_ids.size(0),BERT_MAX_SEQ_LENGTH,dtype=torch.long).to(self.device)

        return_dict=return_dict if return_dict is not None else self.config.use_return_dict
        ret=super().forward(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return ret

class ImageBertForMultipleChoice(BertPreTrainedModel):
    """
    ImageBertModelのトップに全結合層をつけたもの
    BertForMultipleChoiceのImageBERT版
    """
    def __init__(
        self,
        config:BertConfig):
        super().__init__(config)

        self.imbert=ImageBertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size,1)

        self.init_weights()

    def initialize_from_pretrained(self,pretrained_model_name_or_path:str,*model_args,**kwargs):
        self.imbert=ImageBertModel.from_pretrained(pretrained_model_name_or_path,*model_args,**kwargs)
        self.imbert.create_sep_embedding(pretrained_model_name_or_path)

    def to(self,device:torch.device):
        super().to(device)

        self.imbert.to(device)
        self.dropout.to(device)
        self.classifier.to(device)

    def forward(
        self,
        input_ids:torch.Tensor, #(N,num_choices,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor,    #(N,num_choices,max_num_rois,4)
        roi_features:torch.Tensor,  #(N,num_choices,max_num_rois,roi_features_dim)
        labels:torch.Tensor,
        max_num_rois:int,   #入力するRoIの最大数
        output_hidden_states:bool=None,
        return_dict:bool=None):
        num_choices=input_ids.size(1)
        input_ids=input_ids.view(-1,input_ids.size(-1)) #(N*num_choices,BERT_MAX_SEQ_LENGTH)
        roi_boxes=roi_boxes.view(-1,roi_boxes.size(-2),roi_boxes.size(-1)) #(N*num_choices,max_num_rois,4)
        roi_features=roi_features.view(-1,roi_features.size(-2),roi_features.size(-1))    #(N*num_choices,max_num_rois,roi_features_dim)

        outputs=self.imbert(
            input_ids,
            roi_boxes,
            roi_features,
            max_num_rois,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_output=outputs[1]

        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)
        reshaped_logits=logits.view(-1,num_choices)

        criterion=nn.CrossEntropyLoss()
        loss=criterion(reshaped_logits,labels)

        if not return_dict:
            output=(reshaped_logits,)+outputs[2:]
            return ((loss,)+output) if loss is not None else output

        ret={
            "loss":loss,
            "logits":reshaped_logits,
            "hidden_states":outputs.hidden_states,
            "attentions":outputs.attentions,
        }
        return ret
