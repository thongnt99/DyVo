from .margin_mse_loss import MarginMSELoss
from .entity_distil_kl_loss import EntityDistilKLLoss
from .aspect_distil_kl_loss import AspectDistilKLLoss
from .sparse_aspect_kl_loss import SparseAspectKLLoss
from .tokent_aspect_kl_loss import TokentAspectKLLoss
from .multiple_negative_loss import MultipleNegativeLoss
from .term_mse_loss import TermMSELoss
from .cross_entropy_loss import CrossEntropyLoss
from .negative_likelihood import NegativeLikelihoodLoss
from .entity_retrieval_loss import EntityRetrievalLoss
from .entity_retrieval_kl_loss import EntityRetrievalKLLoss
from .distil_kl_loss import DistilKLLoss
from .dense_sparse_distil_kl_loss import DenseSparseDistilKLLoss
from abc import ABC
from torch import nn, Tensor
import torch
