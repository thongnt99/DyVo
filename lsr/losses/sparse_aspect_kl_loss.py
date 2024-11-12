from torch import nn
import torch
from lsr.losses.common import Loss, dot_product, num_non_zero, num_entities


def sparse_dot_product(q_ent_ids, q_ent_mask, q_tok_ids, q_ent_logits, d_ent_ids, d_ent_mask,  d_tok_ids, d_ent_logits):
    batch_size = q_ent_ids.size(0)
    entity_match_mask = (q_ent_ids.unsqueeze(-1) == d_ent_ids.unsqueeze(-2))
    tok_match_mask = (q_tok_ids.unsqueeze(-1) ==
                      d_tok_ids.unsqueeze(-2))
    batch_indices, q_tok_indices, d_tok_indices = tok_match_mask.nonzero(
        as_tuple=True)
    d_match_ent_indices = entity_match_mask.float().argmax(dim=-1)
    d_ent_logits = d_ent_logits[torch.arange(
        batch_size).unsqueeze(-1), :,  d_match_ent_indices].transpose(2, 1)
    d_ent_mask = d_ent_mask[torch.arange(
        batch_size).unsqueeze(-1), d_match_ent_indices]
    d_ent_ids = d_ent_ids[torch.arange(
        batch_size).unsqueeze(-1), d_match_ent_indices]
    mask = q_ent_mask * d_ent_mask * (q_ent_ids == d_ent_ids).float()
    mask = mask[batch_indices]
    q_ent_logits = q_ent_logits[batch_indices,
                                q_tok_indices]
    d_ent_logits = d_ent_logits[batch_indices,
                                d_tok_indices]
    interaction = (q_ent_logits * d_ent_logits * mask)
    if interaction.dim() > 1:
        interaction = interaction.sum(dim=-1)
    scores = torch.zeros(batch_size, device=q_ent_ids.device)
    scores = scores.scatter_add(0, batch_indices, interaction)
    return scores


if __name__ == "__main__":
    d_ent_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])
    d_ent_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    q_ent_ids = torch.tensor([[3, 4, 7, 8], [0, 5, 1, 1]])
    q_ent_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    d_tok_ids = torch.tensor([[100, 101], [105, 105]])
    q_tok_ids = torch.tensor([[100, 101, 200], [105, 105, 800]])
    d_ent_logits = torch.ones(2, 2, 5)
    q_ent_logits = torch.ones(2, 3, 4)
    scores = sparse_dot_product(q_ent_ids, q_ent_mask, q_tok_ids, q_ent_logits,
                                d_ent_ids, d_ent_mask, d_tok_ids, d_ent_logits)
    print(scores)
    # BATCH_SIZE X q_len x d_len

    # batch_size x q_len
    # batch_size x d_len
    # entity_match_mask = q_ent_ids.unsqueeze(
    #     -1) == d_ent_ids.unsqueeze(1)
    # importance_match = q_ent_logits[:, 0, :].unsqueeze(-1) * \
    #     d_ent_logits[:, 0, :].unsqueeze(1)
    # q_aspect_toks = q_tok_ids[:, 1:]
    # d_aspect_toks = d_tok_ids[:, 1:]
    # aspect_match_mask = q_aspect_toks.unsqueeze(
    #     -1) == d_aspect_toks.unsqueeze(-2)  # batch_size x q_len x d_len
    # q_aspect_scores = q_ent_logits[:, 1:, :].transpose(
    #     2, 1)  # batch_size, num_q_entities, q_len
    # d_aspect_scores = d_ent_logits[:, 1:, :].transpose(
    #     2, 1)  # batch_size, num_d_entities, d_len
    # aspect_match = q_aspect_scores.unsqueeze(
    #     -2).unsqueeze(-1) * d_aspect_scores.unsqueeze(1).unsqueeze(-2)  # batch_size, num_q_entities, num_d_entities , q_len, d_len
    # aspect_match = (
    #     aspect_match * aspect_match_mask.unsqueeze(1).unsqueeze(1)).max(dim=-1).values.sum(dim=-1)
    # print(importance_match.size())
    # print(aspect_match.size())
    # print(entity_match_mask.size())
    # interaction = importance_match * aspect_match * entity_match_mask
    # dot_product = interaction.sum(dim=-1).sum(dim=-1)
    # return dot_product


class SparseAspectKLLoss(Loss):
    """
    KL divergence loss for distillation from a teacher model (T) to a student model (S).
    KLLoss(q, p1, p2) = KL(normalize([S(q,p1), S(q,p2)]), normalize([T(q,p1), T(q,p2)])).
    """

    def __init__(self, q_regularizer=None, d_regularizer=None, normalize="softmax") -> None:
        """
        Constructing DistilKLLoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(SparseAspectKLLoss, self).__init__(q_regularizer, d_regularizer)
        self.loss = torch.nn.KLDivLoss(reduction="none")
        self.normalize = normalize

    def forward(self, q_reps, p_reps, n_reps, labels):
        """
        Calculating the KL over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size*2 x vocab_size).
            The number of documents needed is twice the number of query as we need a pair of documents for each query to calculate the margin.
            Documents in even positions (0, 2, 4...) are positive (relevant) documents, documents in odd positions (1, 3, 5...) are negative (non-relvant) documents.
        labels: torch.Tensor
            Teacher's margin between positive and negative documents. labels[i] = teacher(q_reps[i], d_reps[i*2]) - teacher(q_reps[i], d_reps[i*2+1])
        Returns
        -------
        tuple (loss, q_reg, d_reg, log)
            a tuple of averaged loss, query regularization, doc regularization and log (for experiment tracking)
        """
        q_ent_ids, q_ent_mask, q_tok_ids, q_ent_logits = q_reps
        p_ent_ids, p_ent_mask, p_tok_ids, p_ent_logits = p_reps
        n_ent_ids, n_ent_mask, n_tok_ids, n_ent_logits = n_reps

        if self.normalize == "softmax":
            teacher_scores = torch.softmax(labels, dim=1)
        else:
            teacher_scores = torch.nn.functional.normalize(labels, dim=1)
        # similarity with negative documents
        p_rel = sparse_dot_product(
            q_ent_ids, q_ent_mask, q_tok_ids, q_ent_logits, p_ent_ids, p_ent_mask, p_tok_ids, p_ent_logits)
        # similarity with positive documents
        n_rel = sparse_dot_product(
            q_ent_ids, q_ent_mask, q_tok_ids, q_ent_logits, n_ent_ids, n_ent_mask,  n_tok_ids, n_ent_logits)
        student_scores = torch.stack([p_rel, n_rel], dim=1)
        student_scores = torch.log_softmax(student_scores, dim=1)
        reg_q_output = torch.tensor(0.0, device=q_ent_ids.device)
        #     if (self.q_regularizer is None)
        #     else self.q_regularizer(q_ent_logits[:, 0, :])
        # )
        reg_d_output = torch.tensor(0.0, device=q_ent_ids.device)
        # if (self.d_regularizer is None)
        # else (self.d_regularizer(p_ent_logits[:, 0, :]) + self.d_regularizer(n_ent_logits[:, 0, :])) / 2
        # )
        kl_loss = self.loss(student_scores, teacher_scores).sum(
            dim=1).mean(dim=0)
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_ent_logits[:, 0, :]),
            "doc length": num_non_zero(p_ent_logits[:, 0, :]),
            "query #entities": num_entities(q_ent_ids, q_ent_logits[:, 0, :]),
            "doc #entities": num_entities(p_ent_ids, p_ent_logits[:, 0, :]),
            "loss_no_reg": kl_loss.detach(),
        }
        return (kl_loss, reg_q_output, reg_d_output, to_log)
