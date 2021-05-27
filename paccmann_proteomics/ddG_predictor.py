import torch
from torch import nn
from transformers import PreTrainedModel, RobertaConfig, RobertaModel


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForddGPrediction(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # [B, 2, L] => [2B, L]
        input_ids_ = self._flatten(input_ids)
        attention_mask_ = self._flatten(attention_mask)

        token_type_ids_ = None
        if token_type_ids is not None:
            token_type_ids_ = self._flatten(token_type_ids)

        position_ids_ = None
        if position_ids is not None:
            position_ids_ = self._flatten(position_ids)

        head_mask_ = None
        if head_mask is not None:
            head_mask_ = self._flatten(head_mask)

        inputs_embeds_ = None
        if inputs_embeds is not None:
            inputs_embeds_ = self._flatten(inputs_embeds)

        outputs = self.roberta(
            input_ids_,
            attention_mask=attention_mask_,
            token_type_ids=token_type_ids_,
            position_ids=position_ids_,
            head_mask=head_mask_,
            inputs_embeds=inputs_embeds_,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        dG_preds = self.classifier(sequence_output)
        dG_preds = torch.reshape(dG_preds, [-1, 2])

        ddG_preds = dG_preds[:, 1] - dG_preds[:, 0]

        loss = nn.MSELoss()(ddG_preds.squeeze(), labels.squeeze())

        return loss, ddG_preds

    def _flatten(self, tensor):
        return torch.reshape(tensor, [-1, *tensor.shape[2:]])
