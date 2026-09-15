# DeepCluster specification

## Paper mapping

Primary source: [Caron et al., *Deep Clustering for Unsupervised Learning of Visual Features*, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf).

The estimator preserves the defining alternating loop:

1. extract one feature vector per sample with the current backbone;
2. apply KMeans to those features to obtain hard pseudo-labels;
3. initialize a classifier for the current pseudo-label space;
4. update the backbone and classifier using multinomial cross-entropy;
5. repeat, then perform a final feature extraction and clustering pass for public predictions.

Inverse-frequency resampling gives each pseudo-label cluster equal expected sampling mass, matching the paper's response to highly imbalanced assignments. Image training applies stochastic augmentation while feature extraction stays deterministic.

## Input and capability contract

`DeepCluster` accepts dense two-dimensional feature matrices and four-dimensional NCHW image arrays. Two-dimensional inputs use a compact MLP backbone. Images use a compact convolutional backbone and horizontal-flip/noise augmentation. `transform` returns learned features; `predict` applies the final KMeans model. DeepCluster produces hard assignments and does not advertise calibrated soft assignments.

## Evidence boundary

The paper used AlexNet, large-scale ImageNet/YFCC100M training, Sobel preprocessing and its full augmentation/optimization recipe. Those scale-specific choices are not reproduced by the default estimator. The compact CNN, optional Gaussian perturbation and scikit-learn KMeans are practical alternatives that make the alternating semantics testable on CPU. Reference configurations document these differences and must not be compared directly with the paper's transfer-learning results.

No source code from the archived upstream implementation is copied into this repository; the implementation is derived from the published algorithm description.
