from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from .entity_emb import BertEntityEmbedding
from transformers import AutoModelForMaskedLM
import torch
from torch import nn
from transformers import PretrainedConfig
from contextlib import nullcontext


class EPICTermImportance(nn.Module):
    """
    EPICTermImportance class
    This module is used in EPIC model to estimate the term importance for each input term
    Paper: https://arxiv.org/abs/2004.14245
    """

    def __init__(self, dim: int = 768) -> None:
        """
        Construct an EPICTermImportance module
        Parameters
        ----------
        dim: int
            dimension of the input vectors
        """
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        # inputs usually has shape: batch_size x seq_length x vector_size
        s = torch.log1p(self.softplus(self.linear(inputs)))
        return s


class EPICDocQuality(nn.Module):
    """
    EpicDocQuality
    This module is used in EPIC model to estimate the doc quality's score. The input vector is usually the [CLS]'s embedding.
    Paper: https://arxiv.org/abs/2004.14245
    """

    def __init__(self, dim: int = 768) -> None:
        """
        Construct an EpicDocquality module
        Parameters
        ----------
        dim: int
            dimension of the input vector
        """
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.sm = nn.Sigmoid()

    def forward(self, inputs):
        """forward function"""
        s = self.sm(self.linear(inputs))
        return s


class TransformerMLMeMLMConfig(PretrainedConfig):
    """
    Configuration for the TransformerMLMSparseEncoder
    """

    model_type = "MLMeMLM"

    def __init__(
        self,
        tf_base_model_name_or_dir: str = "distilbert-base-uncased",
        pool: str = "max",
        activation: str = "relu",
        norm: str = "log1p",
        term_importance: str = "no",
        entity_dim: int = 300,
        entity_weight: float = 0.01,
        doc_quality: str = "no",
        entity_bias_term: bool = False,
        entity_cls: bool = False,
        ** kwargs,
    ):
        """
        Construct an instance of TransformerMLMConfig
        Paramters
        ---------
        tf_base_model_name_or_dir: str
            name/path of the pretrained weights (HuggingFace) for initializing the masked language model's backbone
        pool: str
            pooling strategy (max, sum)
        activation: str
            activation function
        norm: str
            weight normalization function
        term_importance: str
            module for estimating term importance. "no" for ormitting this component
        doc_quality: str
        """
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.pool = pool
        self.activation = activation
        self.norm = norm
        self.term_importance = term_importance
        self.doc_quality = doc_quality
        self.entity_dim = entity_dim
        self.entity_cls = entity_cls
        self.entity_bias_term = entity_bias_term
        self.entity_weight = entity_weight
        super().__init__(**kwargs)


class TransformerMLMeMLMSparseEncoder(SparseEncoder):
    """
    TransformerMLMSparseEncoder adds a pooling (e.g., max) on top of masked language model's logits.
    """

    config_class = TransformerMLMeMLMConfig

    def __init__(self, config: TransformerMLMeMLMConfig = TransformerMLMeMLMConfig(), ent_emb_model=None):
        super(SparseEncoder, self).__init__(config)
        self.model = AutoModelForMaskedLM.from_pretrained(
            config.tf_base_model_name_or_dir
        )
        # self.activation = FunctionalFactory.get(config.activation)
        self.pool = PoolingFactory.get(config.pool)
        if config.term_importance == "no":
            self.term_importance = functional.AllOne()
        elif config.term_importance == "epic":
            self.term_importance = EPICTermImportance()
        if config.doc_quality == "no":
            self.doc_quality = functional.AllOne()
        elif config.doc_quality == "epic":
            self.doc_quality = EPICDocQuality()
        # self.norm = FunctionalFactory.get(config.norm)
        if config.entity_dim != self.model.config.hidden_size:
            self.entity_proj = nn.Linear(config.entity_dim, self.model.config.hidden_size)
        self.tok_weight = nn.Parameter(torch.tensor(0.5))
        self.ent_weight = nn.Parameter(torch.tensor(self.config.entity_weight))
        self.ent_emb_model = ent_emb_model
        if self.config.entity_bias_term:
            self.entity_bias = nn.Parameter(torch.tensor(0.0))

    def get_hidden_states(self, input_ids, attention_mask, special_tokens_mask=None, token_type_ids=None):
        with torch.no_grad():
            if token_type_ids is not None:
                output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            else:
                output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_states = output.hidden_states[-1]
        return last_hidden_states

    def get_entity_embs(self, entity_ids):
        entity_embs = self.ent_emb_model(entity_ids)
        return entity_embs

    def forward(self, input_ids, attention_mask, special_tokens_mask,  token_type_ids=None, entity_ids=None, entity_masks=None):
        assert entity_ids is not None and entity_masks is not None
        if token_type_ids is not None:
            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        else:
            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # get last_hidden_states
        last_hidden_states = output.hidden_states[-1]
        ##############################################################
        #                      ENTITY SCORES                         #
        ##############################################################
        entity_embs = self.ent_emb_model(entity_ids)
        if self.config.entity_dim != self.model.config.hidden_size:
            entity_embs = self.entity_proj(entity_embs)
        if self.config.entity_cls:
            entity_logits = last_hidden_states[:, 0, :].unsqueeze(
                1) @ entity_embs.transpose(2, 1) * self.ent_weight
        else:
            entity_logits = last_hidden_states @ entity_embs.transpose(
                2, 1) * self.ent_weight
        if self.config.entity_bias_term:
            entity_logits = entity_logits + self.entity_bias
        # Step3. remove negative weights, log scale and max pool
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

        # Step4. max pooling over input token sequence
        entity_weights = entity_logits.max(dim=1).values
        ##############################################################
        #                      WORD   SCORES                         #
        ##############################################################
        term_scores = self.term_importance(last_hidden_states)
        # get cls_tokens: bs x hidden_dim
        cls_toks = output.hidden_states[-1][:, 0, :]
        doc_scores = self.doc_quality(cls_toks)
        # remove padding tokens and special tokens
        logits = (
            output.logits
            * attention_mask.unsqueeze(-1)
            * (1 - special_tokens_mask).unsqueeze(-1)
            * term_scores
        )
        # norm default: log(1+x)
        logits = torch.log1p(torch.relu(logits))
        # (default: max) pooling over sequence tokens
        lex_weights = self.pool(logits) * doc_scores
        max_non_zero = (lex_weights > 0).sum(dim=1).max()
        tok_weights, tok_ids = lex_weights.topk(k=max_non_zero, dim=1)
        ##############################################################
        #                COMBINE ENTITY + WORD SCORES                #
        ##############################################################
        tok_ids = torch.cat([tok_ids, entity_ids], dim=1)
        tok_weights = torch.cat([tok_weights, entity_weights], dim=1)
        return tok_ids, tok_weights, []
