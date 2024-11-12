from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils.functional import FunctionalFactory
from lsr.utils.sparse_rep import SparseRep
from .entity_emb import BertEntityEmbedding
from transformers import AutoModel
import torch
from torch import nn
from transformers import PretrainedConfig
from contextlib import nullcontext


class TransformerMLPeBINConfig(PretrainedConfig):
    """
    Configuration for TransformerMLPSparseEncoder
    """

    model_type = "MLPeBIN"

    def __init__(
        self,
        tf_base_model_name_or_dir="distilbert-base-uncased",
        activation="relu",
        norm="log1p",
        entity_dim=300,
        only_entity: bool = False,
        aspect_vector: bool = False,
        aspect_dim: int = 16,
        sparse_aspect: bool = False,
        freeze_base: bool = False,
        scale=1.0,
        **kwargs,
    ):
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.activation = activation
        self.norm = norm
        self.scale = scale
        self.entity_dim = entity_dim
        self.only_entity = only_entity
        self.aspect_vector = aspect_vector
        self.aspect_dim = aspect_dim
        self.sparse_aspect = sparse_aspect
        self.freeze_base = freeze_base
        super().__init__(**kwargs)


class TransformerMLPeBINSparseEncoder(SparseEncoder):
    """
    MLP on top of Transformer layers
    """

    config_class = TransformerMLPeBINConfig

    def __init__(self, config: TransformerMLPeBINConfig = TransformerMLPeBINConfig()):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(
            config.tf_base_model_name_or_dir)
        if config.freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.activation = FunctionalFactory.get(config.activation)
        self.norm = FunctionalFactory.get(config.norm)
        # self.linear.apply(self._init_weights)
        self.scale = nn.Parameter(torch.tensor(config.scale))
        self.tok_weight = nn.Parameter(torch.tensor(0.5))
        self.ent_weight = nn.Parameter(torch.tensor(0.5))
        # if config.entity_dim != self.model.config.hidden_size:
        self.entity_projection = nn.Sequential(
            nn.Linear(config.entity_dim, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size)
        )
        self.ent_emb_model = BertEntityEmbedding()

    def forward(self, input_ids, special_tokens_mask, attention_mask, token_type_ids=None, entity_ids=None, entity_masks=None,  to_scale=False):
        if token_type_ids:
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
        # if self.config.entity_dim != self.model.config.hidden_size:
        entity_embs = self.entity_projection(entity_embs)
        # Step2. dot products with entity embeddings (mlm)
        entity_logits = last_hidden_states @ entity_embs.transpose(2, 1)
        # Step3. remove negative weights, log scale and max pool
        entity_logits = torch.log1p(
            torch.relu(entity_logits)) * attention_mask.unsqueeze(-1) * entity_masks.unsqueeze(-2)
        # Step4. multiply with a priors or padding mask
        entity_weights = torch.ones_like(entity_ids).float() * entity_masks
        # entity_logits.max(dim=1).values
        if self.config.only_entity:
            if not self.config.aspect_vector:
                return entity_ids, entity_weights, []
            elif self.config.sparse_aspect:
                return entity_ids, entity_masks, input_ids, entity_logits
            else:
                # entity_logit = BS X SEQ_LEN X ENT_LEN
                attentions = torch.softmax(
                    entity_logits.transpose(2, 1), dim=-1)
                # entity_aspects = self.aspect_projection(last_hidden_states)
                entity_aspects = torch.relu(
                    torch.bmm(attentions, last_hidden_states))
                return entity_ids, entity_weights, entity_aspects
        ##############################################################
        #                      WORD   SCORES                         #
        ##############################################################
        tok_weights = self.linear(
            last_hidden_states).squeeze(-1)  # bs x len x 1
        tok_weights = (
            self.norm(self.activation(tok_weights))
            * attention_mask
            * (1 - special_tokens_mask)
        )
        if to_scale:
            tok_weights = tok_weights * self.scale
        ##############################################################
        #                COMBINE ENTITY + WORD SCORES                #
        ##############################################################
        if self.config.aspect_vector:
            if self.config.sparse_aspect:
                return input_ids, tok_weights*self.tok_weight, entity_ids, input_ids, entity_masks, entity_logits*self.ent_weight
            else:
                raise Exception("Not implemented yet")
        else:
            input_ids = torch.cat([input_ids, entity_ids], dim=1)
            tok_weights = torch.cat([tok_weights, entity_weights], dim=1)
            return input_ids, tok_weights, []
