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


class ApproximateContrastiveLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pos_ent_ids, pos_ent_weights, topk_ent_ids, topk_ent_weights):
        # filter-out all sample without annotated entities
        sample_mask = ((pos_ent_weights != 0).sum(dim=1) > 0)
        if sample_mask.sum() == 0:
            return 0
        pos_ent_ids = pos_ent_ids[sample_mask]
        pos_ent_weights = pos_ent_weights[sample_mask]
        topk_ent_ids = topk_ent_ids[sample_mask]
        topk_ent_weights = topk_ent_weights[sample_mask]

        topk_labels = torch.zeros_like(topk_ent_ids)
        pos_labels = torch.ones_like(pos_ent_ids) * (pos_ent_weights != 0)
        logits = torch.cat([pos_ent_weights, topk_ent_weights], dim=1)
        labels = torch.cat([pos_labels, topk_labels], dim=1)
        labels = labels / labels.sum(dim=-1).unsqueeze(-1)
        loss = self.ce(logits, labels)
        return loss
        # # false negative entities (positive entities that are in the topk)
        # fp_ent_mask = (topk_ent_ids.unsqueeze(-1) ==
        #                pos_ent_ids.unsqueeze(1))
        # these entities appear twice
        # topk_labels[fp_ent_mask.sum(dim=-1) > 0] = 0.5
        # pos_labels[fp_ent_mask.sum(dim=1) > 0] = 0.5
        # #
        # pos_labels = pos_labels * (pos_ent_weights != 0)
        # logits =
        # return loss


class EntityRetrievalKLLoss(Loss):
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
        super(EntityRetrievalKLLoss, self).__init__(
            q_regularizer, d_regularizer)
        self.loss = torch.nn.KLDivLoss(reduction="none")
        self.ent_loss = ApproximateContrastiveLoss()
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
        q_tok_ids, q_tok_weights, q_ent_ids, q_ent_weights, q_topk_ent_ids, q_topk_ent_weights = q_reps
        p_tok_ids, p_tok_weights, p_ent_ids, p_ent_weights, p_topk_ent_ids, p_topk_ent_weights = p_reps
        n_tok_ids, n_tok_weights, n_ent_ids, n_ent_weights, n_topk_ent_ids, n_topk_ent_weights = n_reps
        if self.normalize == "softmax":
            teacher_scores = torch.softmax(labels, dim=1)
        else:
            teacher_scores = torch.nn.functional.normalize(
                labels, dim=1)
        # similarity with negative documents
        p_rel = sparse_dot_product(
            q_tok_ids, q_tok_weights, p_tok_ids, p_tok_weights)
        # similarity with positive documents
        n_rel = sparse_dot_product(
            q_tok_ids, q_tok_weights, n_tok_ids, n_tok_weights)
        student_scores = torch.stack([p_rel, n_rel], dim=1)
        student_scores = torch.log_softmax(student_scores, dim=1)
        if self.reg_ent:
            reg_q_output = (
                torch.tensor(0.0, device=q_tok_ids.device)
                if (self.q_regularizer is None)
                else self.q_regularizer(q_tok_weights)
            )
            reg_d_output = (
                torch.tensor(0.0, device=p_tok_weights.device)
                if (self.d_regularizer is None)
                else (self.d_regularizer(p_tok_weights) + self.d_regularizer(n_tok_weights)) / 2
            )
        else:
            reg_q_output = (
                torch.tensor(0.0, device=q_tok_ids.device)
                if (self.q_regularizer is None)
                else self.q_regularizer(q_tok_weights, tok_ids=q_tok_ids)
            )
            reg_d_output = (
                torch.tensor(0.0, device=p_tok_weights.device)
                if (self.d_regularizer is None)
                else (self.d_regularizer(p_tok_weights, tok_ids=p_tok_ids) + self.d_regularizer(n_tok_weights, tok_ids=n_tok_ids)) / 2
            )
        if self.anti_zero:
            reg_q_output += 1.0 / ((q_tok_weights *
                                   (q_tok_ids >= 30522)).sum()**2)
            reg_d_output += 0.5 / ((p_tok_weights *
                                   (p_tok_ids >= 30522)).sum()**2) + 0.5/(n_tok_weights * (n_tok_ids >= 30522)).sum()**2
        kl_loss = self.loss(student_scores, teacher_scores).sum(
            dim=1).mean(dim=0)
        ent_loss = self.ent_loss(
            q_ent_ids, q_ent_weights, q_topk_ent_ids, q_topk_ent_weights) + self.ent_loss(p_ent_ids, p_ent_weights, p_topk_ent_ids, p_topk_ent_weights) + self.ent_loss(n_ent_ids, n_ent_weights, n_topk_ent_ids, n_topk_ent_weights)
        ent_loss = ent_loss / 3
        loss = kl_loss + ent_loss
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_tok_weights),
            "doc length": num_non_zero(p_tok_weights),
            "query #entities": num_entities(q_tok_ids, q_tok_weights),
            "doc #entities": num_entities(p_tok_ids, p_tok_weights),
            "loss_no_reg": kl_loss.detach(),
            "ent_loss": ent_loss.detach()
        }
        return (loss, reg_q_output, reg_d_output, to_log)
