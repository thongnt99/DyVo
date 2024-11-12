from torch import nn
import torch
from lsr.losses.common import Loss, dot_product, num_non_zero, num_entities


def sparse_dot_product(q_tok_ent_ids, q_tok_ent_weights, d_tok_ent_ids, d_tok_ent_weights):
    # batch_size x q_len
    # batch_size x d_len
    exact_match_mask = q_tok_ent_ids.unsqueeze(
        -1) == d_tok_ent_ids.unsqueeze(1)
    interaction = q_tok_ent_weights.unsqueeze(-1) * \
        d_tok_ent_weights.unsqueeze(1) * exact_match_mask
    dot_product = interaction.sum(dim=-1).sum(dim=-1)
    return dot_product


class TripletLoss(Loss):
    """
    """

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        """
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(TripletLoss, self).__init__(q_regularizer, d_regularizer)
        self.ce = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, q_reps, p_reps, n_reps):
        """
        """
        q_tok_ent_ids, q_tok_ent_weights = q_reps
        p_tok_ent_ids, p_tok_ent_weights = p_reps
        n_tok_ent_ids, n_tok_ent_weights = n_reps
        # similarity with negative documents
        p_rel = sparse_dot_product(
            q_tok_ent_ids, q_tok_ent_weights, p_tok_ent_ids, p_tok_ent_weights)
        # similarity with positive documents
        n_rel = sparse_dot_product(
            q_tok_ent_ids, q_tok_ent_weights, n_tok_ent_ids, n_tok_ent_weights)
        student_scores = torch.stack([p_rel, n_rel], dim=1)
        reg_q_output = (
            torch.tensor(0.0, device=q_tok_ent_ids.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_tok_ent_weights)
        )
        reg_d_output = (
            torch.tensor(0.0, device=p_tok_ent_weights.device)
            if (self.d_regularizer is None)
            else (self.d_regularizer(p_tok_ent_weights) + self.d_regularizer(n_tok_ent_weights)) / 2
        )
        labels = torch.zeros(p_rel.size(
            0), device=p_rel.device, dtype=torch.long)
        ce_loss = self.ce(student_scores, labels)
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_tok_ent_weights),
            "doc length": num_non_zero(p_tok_ent_weights),
            "query #entities": num_entities(q_tok_ent_ids, q_tok_ent_weights),
            "doc #entities": num_entities(p_tok_ent_ids, p_tok_ent_weights),
            "loss_no_reg": ce_loss.detach(),
        }
        return (ce_loss, reg_q_output, reg_d_output, to_log)
