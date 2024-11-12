from torch import nn
import torch
from lsr.losses.common import Loss, dot_product, num_non_zero, num_entities


class EntityRetrievalLoss(Loss):
    """
    EntityRetrievalLoss for adapting entity embeddings 
    """

    def __init__(self) -> None:
        """
        Initialize cross-entropy
        """
        super(EntityRetrievalLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, hidden_states, entity_embs):
        """
        """
        # hidden_states:  batch_size x seq_len x 768
        # entity_embs: batch_size x 2 x 768
        # => batch_size x seq_len x 2
        logits = hidden_states @ entity_embs.transpose(2, 1)
        logits = logits.max(dim=1).values  # batch_size x 2
        labels = [0] * logits.size(0)
        loss = self.ce_loss(logits, labels)
        return loss
