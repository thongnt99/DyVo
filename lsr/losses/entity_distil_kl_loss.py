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


class EntityDistilKLLoss(Loss):
    """
    KL divergence loss for distillation from a teacher model (T) to a student model (S).
    KLLoss(q, p1, p2) = KL(normalize([S(q,p1), S(q,p2)]), normalize([T(q,p1), T(q,p2)])).
    """

    def __init__(self, q_regularizer=None, d_regularizer=None, normalize="softmax", reg_ent=True, anti_zero=False) -> None:
        """
        Constructing DistilKLLoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(EntityDistilKLLoss, self).__init__(q_regularizer, d_regularizer)
        self.loss = torch.nn.KLDivLoss(reduction="none")
        self.normalize = normalize
        self.reg_ent = reg_ent
        self.anti_zero = anti_zero

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
        q_tok_ent_ids, q_tok_ent_weights, _ = q_reps
        p_tok_ent_ids, p_tok_ent_weights, _ = p_reps
        n_tok_ent_ids, n_tok_ent_weights, _ = n_reps
        # assert q_tok_ent_ids.size(0) == p_tok_ent_ids.size(
        #     0) == n_tok_ent_ids.size(0) == labels.size(0)
        # assert q_tok_ent_ids.size(1) == q_tok_ent_weights.size(1)
        # assert p_tok_ent_ids.size(1) == p_tok_ent_weights.size(1)
        # assert n_tok_ent_ids.size(1) == n_tok_ent_weights.size(1)
        if self.normalize == "softmax":
            teacher_scores = torch.softmax(labels, dim=1)
        else:
            teacher_scores = torch.nn.functional.normalize(
                labels, dim=1)
        # similarity with negative documents
        p_rel = sparse_dot_product(
            q_tok_ent_ids, q_tok_ent_weights, p_tok_ent_ids, p_tok_ent_weights)
        # similarity with positive documents
        n_rel = sparse_dot_product(
            q_tok_ent_ids, q_tok_ent_weights, n_tok_ent_ids, n_tok_ent_weights)
        student_scores = torch.stack([p_rel, n_rel], dim=1)
        student_scores = torch.log_softmax(student_scores, dim=1)
        if self.reg_ent:
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
        else:
            reg_q_output = (
                torch.tensor(0.0, device=q_tok_ent_ids.device)
                if (self.q_regularizer is None)
                else self.q_regularizer(q_tok_ent_weights, tok_ids=q_tok_ent_ids)
            )
            reg_d_output = (
                torch.tensor(0.0, device=p_tok_ent_weights.device)
                if (self.d_regularizer is None)
                else (self.d_regularizer(p_tok_ent_weights, tok_ids=p_tok_ent_ids) + self.d_regularizer(n_tok_ent_weights, tok_ids=n_tok_ent_ids)) / 2
            )
        if self.anti_zero:
            reg_q_output += 1.0 / ((q_tok_ent_weights *
                                   (q_tok_ent_ids >= 30522)).sum()**2)
            reg_d_output += 0.5 / ((p_tok_ent_weights *
                                   (p_tok_ent_ids >= 30522)).sum()**2) + 0.5/(n_tok_ent_weights * (n_tok_ent_ids >= 30522)).sum()**2
        kl_loss = self.loss(student_scores, teacher_scores).sum(
            dim=1).mean(dim=0)
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
            "loss_no_reg": kl_loss.detach(),
        }
        return (kl_loss, reg_q_output, reg_d_output, to_log)
