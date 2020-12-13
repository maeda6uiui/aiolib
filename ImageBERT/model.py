import torch
import torch.nn as nn
from transformers import(
    BertConfig,
    BertPreTrainedModel
)

from imagebert.model import ImageBertModel

class ImageBertForMultipleChoice(BertPreTrainedModel):
    """
    ImageBertModelのトップに全結合層をつけたもの
    BertForMultipleChoiceのImageBERT版
    """
    def __init__(self,config:BertConfig):
        super().__init__(config)

        self.imbert=ImageBertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size,1)

        self.init_weights()

    def setup_image_bert(self,pretrained_model_name_or_path,*model_args,**kwargs):
        """
        パラメータを事前学習済みのモデルから読み込んでImageBERTのモデルを作成する。
        """
        self.imbert=ImageBertModel.create_from_pretrained(pretrained_model_name_or_path,*model_args,**kwargs)

    def to(self,device:torch.device):
        super().to(device)

        self.imbert.to(device)
        self.dropout.to(device)
        self.classifier.to(device)

    def forward(
        self,
        input_ids:torch.Tensor, #(N,num_choices,BERT_MAX_SEQ_LENGTH)
        labels:torch.Tensor,    #(N)
        token_type_ids:torch.Tensor=None,   #(N,num_choices,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor=None,    #(N,num_choices,max_num_rois,4)
        roi_features:torch.Tensor=None,  #(N,num_choices,max_num_rois,roi_features_dim)
        output_hidden_states:bool=None,
        return_dict:bool=None,
        use_roi_seq_position:bool=False):
        device=self.classifier.weight.device

        input_ids=input_ids.to(device)
        labels=labels.to(device)
        if token_type_ids is not None:
            token_type_ids=token_type_ids.to(device)
        if roi_boxes is not None:
            roi_boxes=roi_boxes.to(device)
        if roi_features is not None:
            roi_features=roi_features.to(device)

        num_choices=input_ids.size(1)
        input_ids=input_ids.view(-1,input_ids.size(-1)) #(N*num_choices,BERT_MAX_SEQ_LENGTH)
        roi_boxes=roi_boxes.view(-1,roi_boxes.size(-2),roi_boxes.size(-1)) #(N*num_choices,max_num_rois,4)
        roi_features=roi_features.view(-1,roi_features.size(-2),roi_features.size(-1))    #(N*num_choices,max_num_rois,roi_features_dim)

        outputs=self.imbert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            roi_boxes=roi_boxes,
            roi_features=roi_features,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_roi_seq_position=use_roi_seq_position
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
