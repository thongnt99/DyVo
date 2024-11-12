from .cls_mlm import TransformerCLSMLPSparseEncoder, TransformerCLSMLMConfig
from .mlm import (
    TransformerMLMSparseEncoder,
    TransformerMLMConfig,
)
from .mlp_entity import MLPEntity, MLPEntityConfig
from .mlm_entity import MLMEntity
from .mlp_ebin import TransformerMLPeBINConfig, TransformerMLPeBINSparseEncoder
from .mlm_emlm import TransformerMLMeMLMConfig, TransformerMLMeMLMSparseEncoder
from .mlp_emlm import TransformerMLPeMLMSparseEncoder, TransformerMLPeMLMConfig
from .mlp import TransformerMLPSparseEncoder, TransformerMLPConfig
from .binary import BinaryEncoder, BinaryEncoderConfig
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, AutoModel
from lsr.models.sparse_encoder import SparseEncoder
import torch
from safetensors.torch import load_file


class DualSparseConfig(PretrainedConfig):
    model_type = "DualEncoder"

    def __init__(self, shared=False, **kwargs):
        self.shared = shared
        super().__init__(**kwargs)


class DualSparseEncoder(PreTrainedModel):
    """
    DualSparseEncoder class that encapsulates encoder(s) for query and document.

    Attributes
    ----------
    shared: bool
        to use a shared encoder for both query/document
    encoder: lsr.models.SparseEncoder
        a shared encoder for encoding both queries and documnets. This encoder is used only if 'shared' is True, otherwise 'None'
    query_encoder: lsr.models.SparseEncoder
        a separate encoder for encoding queries, only if 'shared' is False
    doc_encoder: lsr.models.SparseEncoder
        a separate encoder for encoding documents, only if 'shared' is False
    Methods
    -------
    from_pretrained(model_dir_or_name: str)
    """

    config_class = DualSparseConfig

    def __init__(
        self,
        query_encoder: SparseEncoder,
        doc_encoder: SparseEncoder = None,
        query_encoder_checkpoint=None,
        doc_encoder_checkpoint=None,
        config: DualSparseConfig = DualSparseConfig(),
    ):
        super().__init__(config)
        self.config = config
        if self.config.shared:
            self.encoder = query_encoder
        else:
            self.query_encoder = query_encoder
            self.doc_encoder = doc_encoder
            if query_encoder_checkpoint is not None:
                if query_encoder_checkpoint.endswith("safetensors"):
                    self.query_encoder.load_state_dict(
                        load_file(query_encoder_checkpoint), strict=False)
                else:
                    self.query_encoder.load_state_dict(
                        torch.load(query_encoder_checkpoint), strict=False)
                # for param in self.query_encoder.parameters():
                #     param.requires_grad = False
            if doc_encoder_checkpoint is not None:
                if doc_encoder_checkpoint.endswith("safetensors"):
                    self.doc_encoder.load_state_dict(
                        load_file(doc_encoder_checkpoint), strict=False)
                else:
                    self.doc_encoder.load_state_dict(
                        torch.load(doc_encoder_checkpoint), strict=False)
                # for param in self.doc_encoder.parameters():
                #     param.requires_grad = False

    def encode_queries(self, to_dense=True, **queries):
        """
        Encode a batch of queries with the query encoder
        Arguments
        ---------
        to_dense: bool
            If True, return the output vectors in dense format; otherwise, return in the sparse format (indices, values)
        queries:
            Input dict with {"input_ids": torch.Tensor, "attention_mask": torch.Tensor , "special_tokens_mask": torch.Tensor }
        """
        if to_dense:
            if self.config.shared:
                return self.encoder(**queries).to_dense(reduce="sum")
            else:
                return self.query_encoder(**queries).to_dense(reduce="sum")
        else:
            if self.config.shared:
                return self.encoder(**queries)
            else:
                return self.query_encoder(**queries)

    def encode_docs(self, to_dense=True, **docs):
        """
        Encode a batch of documents with the document encoder
        """
        if to_dense:
            if self.config.shared:
                return self.encoder(**docs).to_dense(reduce="amax")
            else:
                return self.doc_encoder(**docs).to_dense(reduce="amax")
        else:
            if self.config.shared:
                return self.encoder(**docs)
            else:
                return self.doc_encoder(**docs)

    def forward(self, loss, queries, doc_groups, labels=None, **kwargs):
        """Compute the loss given (queries, docs, labels)"""
        # if "entity_ids" in queries:
        q_reps = self.encode_queries(**queries, to_dense=False)
        docs_groups = [self.encode_docs(
            **doc_group, to_dense=False) for doc_group in doc_groups]
        # else:
        #     q_reps = self.encode_queries(**queries)
        #     docs_batch_rep = self.encode_docs(**docs_batch)
        if labels is None:
            output = loss(q_reps, *docs_groups)
        else:
            output = loss(q_reps, *docs_groups, labels=labels)
        return output

    def save_pretrained(self, model_dir):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir)
        if self.config.shared:
            self.encoder.save_pretrained(model_dir + "/shared_encoder")
        else:
            self.query_encoder.save_pretrained(model_dir + "/query_encoder")
            self.doc_encoder.save_pretrained(model_dir + "/doc_encoder")

    @classmethod
    def from_pretrained(cls, model_dir_or_name):
        """Load query and doc encoder from a directory"""
        config = DualSparseConfig.from_pretrained(model_dir_or_name)
        if config.shared:
            shared_encoder = AutoModel.from_pretrained(
                model_dir_or_name + "/shared_encoder"
            )
            return cls(shared_encoder, config=config)
        else:
            query_encoder = AutoModel.from_pretrained(
                model_dir_or_name + "/query_encoder"
            )
            doc_encoder = AutoModel.from_pretrained(
                model_dir_or_name + "/doc_encoder")
            return cls(query_encoder, doc_encoder, config=config)


class EntityAdapter(DualSparseEncoder):
    def forward(self, loss, queries, query_entities, docs, doc_entities, **kwargs):
        query_hidden_states = self.query_encoder.get_hidden_states(**queries)
        query_ent_embs = self.query_encoder.get_entity_embs(**query_entities)
        doc_hidden_states = self.doc_encoder.get_hidden_states(**docs)
        doc_ent_embs = self.doc_encoder.get_entity_embs(**doc_entities)
        query_loss = loss(query_hidden_states, query_ent_embs)
        doc_loss = loss(doc_hidden_states, doc_ent_embs)
        return (query_loss + doc_loss)/2.0


AutoConfig.register("BINARY", BinaryEncoderConfig)
AutoModel.register(BinaryEncoderConfig, BinaryEncoder)
AutoConfig.register("MLP", TransformerMLPConfig)
AutoModel.register(TransformerMLPConfig, TransformerMLPSparseEncoder)
AutoConfig.register("MLM", TransformerMLMConfig)
AutoModel.register(TransformerMLMConfig, TransformerMLMSparseEncoder)
AutoConfig.register("CLS_MLM", TransformerCLSMLMConfig)
AutoModel.register(TransformerCLSMLMConfig, TransformerCLSMLPSparseEncoder)
AutoConfig.register("MLPeMLM", TransformerMLPeMLMConfig)
AutoModel.register(TransformerMLPeMLMConfig, TransformerMLPeMLMSparseEncoder)
AutoConfig.register("MLMeMLM", TransformerMLMeMLMConfig)
AutoModel.register(TransformerMLMeMLMConfig, TransformerMLMeMLMSparseEncoder)
AutoConfig.register("MLPeBIN", TransformerMLPeBINConfig)
AutoModel.register(TransformerMLPeBINConfig, TransformerMLPeBINSparseEncoder)
