import logging
import torch
import torch.nn as nn
from transformers import(
    BertConfig,
    BertModel
)

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

BERT_MAX_SEQ_LENGTH=512

class ImageBertModel(BertModel):
    """
    ImageBERTのモデル
    """
    def __init__(
        self,
        config:BertConfig,
        add_pooling_layer:bool=True,
        roi_features_dim:int=1024,
        max_num_rois:int=100,
        image_width:int=256,
        image_height:int=256,
        logger:logging.Logger=default_logger):
        super().__init__(config,add_pooling_layer=add_pooling_layer)
        self.logger=logger

        self.roi_features_dim=roi_features_dim
        self.max_num_rois=max_num_rois

        self.fc_roi_boxes=nn.Linear(5,config.hidden_size)
        self.fc_roi_features=nn.Linear(roi_features_dim,config.hidden_size)

        self.position_ids=torch.empty(BERT_MAX_SEQ_LENGTH,dtype=torch.long)
        for i in range(BERT_MAX_SEQ_LENGTH):
            self.position_ids[i]=i
        self.text_token_type_ids=torch.zeros(BERT_MAX_SEQ_LENGTH,dtype=torch.long)
        self.roi_token_type_ids=torch.ones(BERT_MAX_SEQ_LENGTH,dtype=torch.long)

        self.wh_tensor=torch.empty(max_num_rois,5)  #(RoIの)Position Embedding作成に使用する。
        for i in range(max_num_rois):
            self.wh_tensor[i,0]=image_width
            self.wh_tensor[i,1]=image_height
            self.wh_tensor[i,2]=image_width
            self.wh_tensor[i,3]=image_height
            self.wh_tensor[i,4]=image_width*image_height

        self.device=torch.device("cpu") #デフォルトではCPU

    def to(self,device:torch.device):
        self.device=device

        super().to(device)

        self.to(device)
        self.fc_roi_boxes.to(device)
        self.fc_roi_features.to(device)
        self.position_ids.to(device)
        self.text_token_type_ids.to(device)
        self.roi_token_type_ids.to(device)
        self.wh_tensor.to(device)

    def __trim_roi_tensors(self,tensor:torch.Tensor)->torch.Tensor:
        """
        各バッチで含まれるRoIの数が異なると処理が面倒なので、max_num_roisに合わせる。
        もしもmax_num_roisよりも多い場合には切り捨て、
        max_num_roisよりも少ない場合には0ベクトルで埋める。

        入力Tensorのサイズ
        roi_boxes: (N,num_rois,x)

        出力Tensorのサイズ
        (N,max_num_rois,x)
        """
        batch_size=tensor.size(0)
        x=tensor.size(-1)
        
        ret=torch.empty(batch_size,self.max_num_rois,x).to(self.device)

        for i in range(batch_size):
            roi_boxes_tmp=tensor[i]
            num_rois=roi_boxes_tmp.size(0)

            #RoIの数が制限よりも多い場合はTruncateする。
            if num_rois>self.max_num_rois:
                roi_boxes_tmp=roi_boxes_tmp[:self.max_num_rois]
            #RoIの数が制限よりも少ない場合は0ベクトルで埋める。
            elif num_rois<self.max_num_rois:
                zeros=torch.zeros(self.max_num_rois-num_rois,x).to(self.device)
                roi_boxes_tmp=torch.cat([roi_boxes_tmp,zeros],dim=0)

            ret[i]=roi_boxes_tmp

        return ret

    def __create_embeddings(
        self,
        input_ids:torch.Tensor,
        roi_boxes:torch.Tensor,
        roi_features:torch.Tensor)->torch.Tensor:
        """
        入力Embeddingを作成する。
        """
        word_embeddings=self.embeddings.word_embeddings
        position_embeddings=self.embeddings.position_embeddings
        token_type_ids_embeddings=self.embeddings.token_type_embeddings
        layer_norm=self.embeddings.LayerNorm
        dropout=self.embeddings.dropout

        v_position_embeddings=position_embeddings(self.position_ids)
        v_text_token_type_ids_embeddings=token_type_ids_embeddings(self.text_token_type_ids)
        v_roi_token_type_ids_embeddings=token_type_ids_embeddings(self.roi_token_type_ids)

        #=== テキストEmbeddingを作成する。===
        v_word_embeddings=word_embeddings(input_ids)

        #=== RoIのEmbeddingを作成する。 ===
        roi_features=self.__trim_roi_tensors(roi_features)  #(N,max_num_rois,roi_features_dim)
        roi_features=roi_features.view(-1,self.roi_features_dim)    #(N*max_num_rois,roi_features_dim)
        roi_features_embeddings=self.fc_roi_features(roi_features)  #(N*max_num_rois,hidden_size)
        roi_features_embeddings=roi_features_embeddings.view(-1,self.max_num_rois,self.roi_features_dim)

        #RoIの座標から(RoIの)Position Embeddingを作成する。
        roi_boxes=self.__trim_roi_tensors(roi_boxes)    #(N,max_num_rois,4)

        roi_position_embeddings=torch.empty(self.max_num_rois,5).to(self.device)
        for i in range(self.max_num_rois):
            x_tl=roi_boxes[i,0]
            y_tl=roi_boxes[i,1]
            x_br=roi_boxes[i,2]
            y_br=roi_boxes[i,3]

            roi_position_embeddings[i,0]=x_tl
            roi_position_embeddings[i,1]=y_tl
            roi_position_embeddings[i,2]=x_br
            roi_position_embeddings[i,3]=y_br
            roi_position_embeddings[i,4]=(x_br-x_tl)*(y_br-y_tl)

        roi_position_embeddings=torch.div(roi_position_embeddings,self.wh_tensor)

        #RoIのPosition Embeddingを射影する。
        roi_position_embeddings=self.fc_roi_boxes(roi_position_embeddings)

        roi_embeddings=roi_features_embeddings+roi_position_embeddings

        #=== テキストEmbeddingとRoI Embeddingを結合する。
        trunc_word_embeddings=v_word_embeddings[:,:BERT_MAX_SEQ_LENGTH-self.max_num_rois,:]
        text_roi_embeddings=torch.cat([trunc_word_embeddings,roi_embeddings],dim=1)

        trunc_text_token_type_ids=v_text_token_type_ids_embeddings[:BERT_MAX_SEQ_LENGTH-self.max_num_rois]
        trunc_roi_token_type_ids=v_roi_token_type_ids_embeddings[BERT_MAX_SEQ_LENGTH-self.max_num_rois:]
        v_token_type_ids_embeddings=torch.cat([trunc_text_token_type_ids,trunc_roi_token_type_ids],dim=0)

        embeddings=text_roi_embeddings+v_position_embeddings+v_token_type_ids_embeddings
        embeddings=layer_norm(embeddings)
        embeddings=dropout(embeddings)

        return embeddings

    def forward(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor,    #(N,number of RoIs,dimension of features)
        roi_features:torch.Tensor,
        output_hidden_states:bool=None,
        return_dict:bool=None):
        embeddings=self.__create_embeddings(input_ids,roi_boxes,roi_features)

        #Todo: とりあえずattention_maskはすべて1
        attention_mask=torch.ones(input_ids.size(0),BERT_MAX_SEQ_LENGTH,dtype=torch.long).to(self.device)

        ret=super().forward(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        return ret
