from torch import nn
import torch
from lsr.losses.common import Loss, cross_dot_product, num_non_zero, num_entities


def sparse_cross_dot_product(q_tok_ent_ids, q_tok_ent_weights, d_tok_ent_ids, d_tok_ent_weights):
    # batch_size x q_len
    # batch_size x d_len
    exact_match_mask = q_tok_ent_ids.unsqueeze(
        1).unsqueeze(-1) == d_tok_ent_ids.unsqueeze(1).unsqueeze(0)
    interaction = q_tok_ent_weights.unsqueeze(
        1).unsqueeze(-1) * \
        d_tok_ent_weights.unsqueeze(1).unsqueeze(0) * exact_match_mask
    dot_product = interaction.sum(dim=-1).sum(dim=-1)
    assert dot_product.size(0) == q_tok_ent_ids.size(0)
    assert dot_product.size(1) == d_tok_ent_ids.size(0)
    return dot_product


class MultipleNegativeLoss(Loss):
    """
    The MultipleNegativeLoss implements the CrossEntropyLoss underneath. There are one positive document and multiple negative documents per query.
    For each query, this loss considers two type of negatives:
        1. The query's own negatives sampled from traning data.
        2. Documents (both positive and negative) from other queries. (in-batch negatives)
    """

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        """
        Constructing MultipleNegativeLoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(MultipleNegativeLoss, self).__init__(
            q_regularizer, d_regularizer)
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, q_reps, *d_group, labels=None):
        """
        Calculating the MultipleNegativeLoss over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size*group_size x vocab_size).
            group_size is the numer of documents (positive & negative) per query. The first document of the group is positive, the rest are negatives.
        Returns
        -------
        tuple (loss, q_reg, d_reg, log)
            a tuple of averaged loss, query regularization, doc regularization and log (for experiment tracking)
        """
        q_tok_ids, q_tok_weights = q_reps
        sim_matrix = []
        for doc in d_group:
            d_tok_ids, d_tok_weights = doc
            qdsim = sparse_cross_dot_product(
                q_tok_ids, q_tok_weights, d_tok_ids, d_tok_weights)
            sim_matrix.append(qdsim)
        sim_matrix = torch.cat(sim_matrix, dim=1)
        # cross_dot_product(q_reps, d_reps)
        reg_q_output = (
            torch.tensor(0.0, device=q_tok_weights.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_tok_weights)
        )
        reg_d_output = (
            torch.tensor(0.0, device=q_tok_weights)
            if (self.d_regularizer is None)
            else torch.tensor([self.d_regularizer(doc[1]) for doc in d_group]).mean()
        )
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        q_num, d_sum = sim_matrix.size()
        assert d_sum % q_num == 0
        labels = torch.arange(0, q_num, device=sim_matrix.device)
        ce_loss = self.ce(sim_matrix, labels)
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_tok_ids),
            "doc length": num_non_zero(d_group[0][1]),
            "query #entities": num_entities(q_tok_ids, q_tok_weights),
            "doc #entities": num_entities(*d_group[0]),

        }
        return (ce_loss, reg_q_output, reg_d_output, to_log)
