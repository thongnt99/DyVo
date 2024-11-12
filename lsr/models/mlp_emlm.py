from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils.functional import FunctionalFactory
from lsr.utils.sparse_rep import SparseRep
from .entity_emb import BertEntityEmbedding
from transformers import AutoModel, AutoModelForMaskedLM
import torch
from torch import nn
from transformers import PretrainedConfig
from contextlib import nullcontext


class TransformerMLPeMLMConfig(PretrainedConfig):
    """
    Configuration for TransformerMLPSparseEncoder
    """

    model_type = "MLPeMLM"

    def __init__(
        self,
        tf_base_model_name_or_dir="distilbert-base-uncased",
        activation="relu",
        norm="log1p",
        entity_dim=300,
        scale=1.0,
        entity_weight: float = 0.01,
        entity_cls: bool = False,
        entity_bias_term: bool = False,
        ** kwargs,
    ):
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.activation = activation
        self.norm = norm
        self.scale = scale
        self.entity_dim = entity_dim
        self.entity_bias_term = entity_bias_term
        self.entity_weight = entity_weight

        super().__init__(**kwargs)


class TransformerMLPeMLMSparseEncoder(SparseEncoder):
    """
    MLP on top of Transformer layers
    """

    config_class = TransformerMLPeMLMConfig

    def __init__(self, config: TransformerMLPeMLMConfig = TransformerMLPeMLMConfig(), ent_emb_model=None):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(
            config.tf_base_model_name_or_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.scale = nn.Parameter(torch.tensor(config.scale))
        self.tok_weight = nn.Parameter(torch.tensor(0.5))
        self.ent_weight = nn.Parameter(torch.tensor(self.config.entity_weight))
        if config.entity_dim != self.model.config.hidden_size:
            self.entity_proj = nn.Linear(config.entity_dim, self.model.config.hidden_size)
        self.ent_emb_model = ent_emb_model
        if self.config.entity_bias_term:
            self.entity_bias = nn.Parameter(torch.tensor(0.0))


    def get_hidden_states(self, input_ids, attention_mask, special_tokens_mask=None, token_type_ids=None):
        with torch.no_grad():
            if token_type_ids is not None:
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask)
            last_hidden_states = output.last_hidden_state
        return last_hidden_states

    def get_entity_embs(self, entity_ids):
        entity_embs = self.ent_emb_model(entity_ids)
        return entity_embs

    def forward(self, input_ids, special_tokens_mask, attention_mask, token_type_ids=None, entity_ids=None, entity_masks=None,  to_scale=False):
        if token_type_ids is not None:
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask)
        last_hidden_states = output.last_hidden_state
       ##############################################################
        #                      ENTITY SCORES                         #
        ##############################################################
        # Step1. hidden state: 768 -> 300
        entity_embs = self.ent_emb_model(entity_ids)
        if self.config.entity_dim != self.model.config.hidden_size:
            entity_embs = self.entity_proj(entity_embs)
        # Step2. dot products with entity embeddings (mlm)
        if self.config.entity_cls:
            entity_logits = last_hidden_states[:, 0, :].unsqueeze(
                0) @ entity_embs.transpose(2, 1) * self.ent_weight
        else:
            entity_logits = last_hidden_states @ entity_embs.transpose(
                2, 1) * self.ent_weight
        if self.config.entity_bias_term:
            entity_logits = entity_logits + self.entity_bias
        # Step3. remove negative weights, log scale
        if self.config.activation == "relu":
            entity_logits = torch.log1p(
                torch.relu(entity_logits)) * entity_masks.unsqueeze(-2)
        elif self.config.activation == "softplus":
            entity_logits = torch.log1p(
                torch.nn.functional.softplus(entity_logits)) * entity_masks.unsqueeze(-2)
        elif self.config.activation == "sigmoid":
            entity_logits = torch.sigmoid(
                entity_logits) * entity_masks.unsqueeze(-2)
        elif self.config.activation == "no":
            entity_logits = entity_logits * entity_masks.unsqueeze(-2)
        # Step4.  max pooling over text input sequence
        entity_weights = entity_logits.max(dim=1).values
        ##############################################################
        #                      WORD   SCORES                         #
        ##############################################################
        tok_weights = self.linear(
            last_hidden_states).squeeze(-1)  # bs x len x 1
        tok_weights = (
            torch.log1p(torch.relu(tok_weights))
            * attention_mask
            * (1 - special_tokens_mask)
        )
        if to_scale:
            tok_weights = tok_weights * self.scale
        ##############################################################
        #                COMBINE ENTITY + WORD SCORES                #
        ##############################################################
        input_ids = torch.cat([input_ids, entity_ids], dim=1)
        tok_weights = torch.cat([tok_weights, entity_weights], dim=1)
        return input_ids, tok_weights, []
