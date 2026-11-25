"""Bayesian Personalized Ranking on user-specific positive/negative scores."""

from torch.nn.functional import softplus


def bpr_loss(positive_scores, negative_scores):
    """Average the pairwise loss for one positive and several negatives per user."""
    return softplus(negative_scores - positive_scores[:, None]).mean()
