import torch
from torch.nn import functional as F

from opendeepclustering.components import (
    GaussianMixturePrior,
    StudentTClustering,
    target_distribution,
)


def test_target_distribution_uses_global_cluster_frequencies():
    q = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    expected_weight = q.square() / q.sum(dim=0)
    expected = expected_weight / expected_weight.sum(dim=1, keepdim=True)
    torch.testing.assert_close(target_distribution(q), expected)
    assert not torch.allclose(
        target_distribution(q),
        torch.cat([target_distribution(q[:1]), target_distribution(q[1:])]),
    )


def test_student_t_assignment_matches_equation():
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]]).numpy()
    layer = StudentTClustering(2, 2, alpha=1.0, initial_centers=centers)
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    distances = torch.tensor([[0.0, 4.0], [1.0, 1.0]])
    numerator = (1.0 + distances).pow(-1.0)
    expected = numerator / numerator.sum(dim=1, keepdim=True)
    torch.testing.assert_close(layer(points), expected)


def test_kl_loss_is_p_log_p_over_q():
    p = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    q = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    expected = (p * (p.log() - q.log())).sum() / len(p)
    actual = F.kl_div(q.log(), p, reduction="batchmean")
    torch.testing.assert_close(actual, expected)


def test_gaussian_mixture_prior_normalizes_and_has_zero_identity_kl():
    prior = GaussianMixturePrior(n_clusters=1, latent_dim=2)
    prior.initialize(weights=[1.0], means=[[0.0, 0.0]], variances=[[1.0, 1.0]])
    mean = torch.zeros(3, 2)
    log_variance = torch.zeros(3, 2)
    kl, responsibilities = prior.expected_kl(mean, log_variance)
    torch.testing.assert_close(kl, torch.zeros(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(responsibilities, torch.ones(3, 1))
