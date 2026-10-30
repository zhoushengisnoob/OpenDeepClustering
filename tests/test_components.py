import pytest
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


def test_gaussian_mixture_prior_parameter_shapes_and_weight_normalization():
    prior = GaussianMixturePrior(n_clusters=4, latent_dim=3)
    assert prior.logits.shape == (4,)
    assert prior.means.shape == (4, 3)
    assert prior.log_variances.shape == (4, 3)

    weights = [0.1, 0.2, 0.3, 0.4]
    means = [[float(i)] * 3 for i in range(4)]
    variances = [[1.0, 2.0, 3.0] for _ in range(4)]
    prior.initialize(weights=weights, means=means, variances=variances)

    # initialize stores log-weights, so softmax must recover the input.
    torch.testing.assert_close(
        torch.as_tensor(prior.weights), torch.tensor(weights), atol=1e-6, rtol=1e-6
    )
    assert float(torch.as_tensor(prior.weights).sum()) == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(
        prior.log_variances.exp(), torch.tensor(variances), atol=1e-5, rtol=1e-5
    )


def test_gaussian_mixture_prior_responsibilities_normalize_per_sample():
    prior = GaussianMixturePrior(n_clusters=3, latent_dim=2)
    prior.initialize(
        weights=[0.5, 0.3, 0.2],
        means=[[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0]],
        variances=[[1.0, 1.0]] * 3,
    )
    z = torch.tensor([[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0], [10.0, 10.0]])

    responsibilities = prior.responsibilities(z)
    assert responsibilities.shape == (4, 3)
    torch.testing.assert_close(
        responsibilities.sum(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
    )
    assert (responsibilities >= 0).all()
    # Each of the first three points sits on a distinct component's mean.
    torch.testing.assert_close(
        responsibilities[:3].argmax(dim=1), torch.tensor([0, 1, 2])
    )

    # expected_kl returns its own responsibilities; they must normalize too.
    _, variational = prior.expected_kl(z, torch.zeros_like(z))
    torch.testing.assert_close(
        variational.sum(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
    )


def test_gaussian_mixture_prior_clamps_degenerate_covariance():
    """A zero-variance GMM component must not produce a non-finite prior.

    scikit-learn can return a near-zero covariance for a component holding
    almost identical points. initialize clamps at 1e-6 so prior_variance never
    reaches the denominator of expected_kl as zero.
    """
    prior = GaussianMixturePrior(n_clusters=2, latent_dim=2)
    prior.initialize(
        weights=[0.5, 0.5],
        means=[[0.0, 0.0], [1.0, 1.0]],
        variances=[[0.0, 0.0], [1e-12, 1.0]],
    )
    assert torch.isfinite(prior.log_variances).all()
    # The clamp floor is 1e-6; allow for float32 log/exp round-trip error.
    assert float(prior.log_variances.exp().min()) == pytest.approx(1e-6, rel=1e-5)

    kl, responsibilities = prior.expected_kl(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.zeros(2, 2)
    )
    assert torch.isfinite(kl).all()
    assert torch.isfinite(responsibilities).all()
    torch.testing.assert_close(
        responsibilities.sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6
    )


def test_gaussian_mixture_prior_rejects_zero_weight_without_underflow():
    """A component the GMM assigned no mass must stay representable."""
    prior = GaussianMixturePrior(n_clusters=2, latent_dim=2)
    prior.initialize(
        weights=[1.0, 0.0],
        means=[[0.0, 0.0], [5.0, 5.0]],
        variances=[[1.0, 1.0]] * 2,
    )
    assert torch.isfinite(prior.logits).all()
    recovered = torch.as_tensor(prior.weights)
    assert float(recovered.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(recovered[1]) >= 0.0
    kl, _ = prior.expected_kl(torch.zeros(1, 2), torch.zeros(1, 2))
    assert torch.isfinite(kl).all()
