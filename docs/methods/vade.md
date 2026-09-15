# VaDE specification

## Paper mapping

Primary source: [Jiang et al., *Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering*, IJCAI 2017](https://www.ijcai.org/proceedings/2017/273).

The generative model follows the paper:

1. choose a cluster, `c ~ Cat(pi)`;
2. draw a latent vector from the cluster-conditional diagonal Gaussian, `z | c ~ N(mu_c, sigma_c² I)`;
3. decode the latent vector into an observation;
4. infer a diagonal-Gaussian `q(z | x)` with an encoder and optimize the evidence lower bound using the reparameterization trick;
5. obtain soft cluster assignments from the Gaussian-mixture posterior.

The loss combines reconstruction likelihood with the analytic expectation of `KL(q(z, c | x) || p(z, c))`. The mixture posterior is normalized in log space. As in the paper, an autoencoder warm-up is followed by Gaussian-mixture initialization in the latent space.

## Input and capability contract

`VaDE` accepts dense two-dimensional feature matrices. It exposes posterior means through `transform`, mixture responsibilities through `soft_assign`/`predict_proba`, hard labels through `predict`, and decoded prior samples through `sample`.

## Evidence boundary

The paper uses Bernoulli reconstruction for MNIST and Gaussian reconstruction for other real-valued datasets. Both are selectable; Gaussian is the API default because arbitrary feature matrices are not guaranteed to lie in `[0, 1]`. The implementation uses a compact MLP and diagonal covariance, and initializes with scikit-learn's GaussianMixture. These preserve the paper's probabilistic structure but are not a claim to reproduce its dataset-specific architecture, learning-rate decay or reported accuracy without the committed reference protocol.

No source code from third-party VaDE repositories is copied into this implementation.
