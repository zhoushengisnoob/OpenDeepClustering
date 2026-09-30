# Dataset sources and local storage

Only this README is versioned under `datasets/`. Downloaded archives, extracted files, prepared NPZ arrays, manifests and local preparation scripts are ignored by Git. A fresh clone contains the documentation, not the datasets.

Run the commands below from the repository root after following the [environment setup](../README.md#installation):

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Sources and splits

| Dataset | Source | Original labeled splits | Labels / features |
| --- | --- | --- | --- |
| MNIST | [Torchvision MNIST loader](https://docs.pytorch.org/vision/0.20/generated/torchvision.datasets.MNIST.html), which downloads the original MNIST files | 60,000 train + 10,000 test | 10 classes; 28×28 grayscale |
| Fashion-MNIST | [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) | 60,000 train + 10,000 test | 10 classes; 28×28 grayscale |
| USPS | [LIBSVM dataset collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps) | 7,291 train + 2,007 test | 10 classes; 256 pixel features |
| CIFAR-10 | [Dataset authors](https://www.cs.toronto.edu/~kriz/cifar.html) | 50,000 train + 10,000 test | 10 classes; 32×32 RGB |
| CIFAR-100 / CIFAR-100-20 | [Dataset authors](https://www.cs.toronto.edu/~kriz/cifar.html) | 50,000 train + 10,000 test | 100 fine classes or 20 superclasses; 32×32 RGB |
| STL-10 | [Stanford dataset page](https://cs.stanford.edu/~acoates/stl10/) | 5,000 train + 8,000 test | 10 classes; 96×96 RGB; an additional 100,000 unlabeled images |
| Reuters-10K | [VaDE authors' prepared file at a pinned commit](https://github.com/slim1017/VaDE/blob/eca52b4b57e32a68f578de7a54eda5d39907b0fe/dataset/reuters10k/reuters10k.mat) | 10,000 documents in this prepared version | 4 classes; 2,000-dimensional TF-IDF features |

The Reuters file is the VaDE author-provided four-class version, pinned to commit `eca52b4b57e32a68f578de7a54eda5d39907b0fe`. Do not assume that every DEC/IDEC derivative uses identical samples or preprocessing. Choose one version and record its identity for all comparisons.

Preserve the original train/test split when downloading. Explicitly state whether an experiment uses train only, train+test, or extra unlabeled data. STL-10's unlabeled samples have no ground-truth labels for ACC/NMI/ARI. CIFAR-100-20 uses the 20 coarse labels, not the 100 fine labels. Cite the dataset authors when reporting results.

## Download image datasets

Torchvision handles the source URLs, archive checksums and extraction. This command explicitly downloads both labeled splits of all six image datasets; STL-10 downloads a large archive that also contains the unlabeled images. For only MNIST, use the shorter command in the root README.

```bash
python - <<'PY'
from torchvision.datasets import MNIST, FashionMNIST, USPS, CIFAR10, CIFAR100, STL10

for cls in (MNIST, FashionMNIST, CIFAR10, CIFAR100):
    for train in (True, False):
        dataset = cls("datasets", train=train, download=True)
        print(cls.__name__, "train" if train else "test", len(dataset))

for train in (True, False):
    dataset = USPS("datasets/USPS", train=train, download=True)
    print("USPS", "train" if train else "test", len(dataset))

for split in ("train", "test"):
    dataset = STL10("datasets", split=split, download=True)
    print("STL10", split, len(dataset))
PY
```

Expected local layout after downloading and any optional preparation:

```text
datasets/
├── README.md                         # the only versioned file
├── MNIST/raw/
├── FashionMNIST/raw/
├── USPS/
├── cifar-10-batches-py/
├── cifar-100-python/
├── stl10_binary/
├── Reuters10K/reuters10k.mat
└── prepared/                         # optional arrays for the NPZ benchmark loader
```

## Download Reuters-10K

The following Linux command downloads the pinned author-provided file and checks its SHA-256 before preparation:

```bash
mkdir -p datasets/Reuters10K
curl --fail --location --retry 3 \
    -H 'Accept: application/vnd.github.raw+json' \
    'https://api.github.com/repos/slim1017/VaDE/contents/dataset/reuters10k/reuters10k.mat?ref=eca52b4b57e32a68f578de7a54eda5d39907b0fe' \
    -o datasets/Reuters10K/reuters10k.mat
printf '%s\n' 'aa0774a824889efe0597f0d42c6a795c99f312f89819afde957746fb7f0588b1  datasets/Reuters10K/reuters10k.mat' | sha256sum --check

python - <<'PY'
from pathlib import Path
import numpy as np
from scipy.io import loadmat

data = loadmat("datasets/Reuters10K/reuters10k.mat")
X = np.asarray(data["X"], dtype=np.float32)
y = data["Y"].ravel().astype(np.int64)
if np.array_equal(np.unique(y), np.arange(1, 5)):
    y = y - 1
assert X.shape == (10000, 2000)
assert np.array_equal(np.unique(y), np.arange(4))
assert np.isfinite(X).all()
Path("datasets/prepared").mkdir(exist_ok=True)
np.savez_compressed("datasets/prepared/Reuters10K.npz", X=X, y=y)
PY
```

The saved features preserve the author's TF-IDF values, converted to float32. No additional normalization is applied.

## Use datasets with the benchmark CLI

The current CLI has built-in loaders for MNIST, synthetic blobs and NPZ arrays. Downloading Fashion-MNIST, USPS, CIFAR or STL-10 does not automatically add a built-in CLI loader. Use the Python API or prepare an NPZ file with:

- `X`: finite sample features or NCHW image arrays.
- `y`: one-dimensional, zero-based class labels.
- Optional `split`: `0` for train and `1` for test, preserving the source split.
- Optional `y_coarse`: the 20 coarse labels for CIFAR-100-20.

For example, prepare Fashion-MNIST without changing pixel values:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from torchvision.datasets import FashionMNIST

train = FashionMNIST("datasets", train=True, download=False)
test = FashionMNIST("datasets", train=False, download=False)
X = np.concatenate([train.data.numpy(), test.data.numpy()])[:, None, :, :]
y = np.concatenate([train.targets.numpy(), test.targets.numpy()])
split = np.concatenate([np.zeros(len(train), dtype=np.uint8),
                        np.ones(len(test), dtype=np.uint8)])
Path("datasets/prepared").mkdir(exist_ok=True)
np.savez_compressed("datasets/prepared/FashionMNIST.npz", X=X, y=y, split=split)
PY
```

A configuration under `configs/benchmarks/` can then contain:

```yaml
dataset:
  kind: npz
  path: ../../datasets/prepared/FashionMNIST.npz
  features_key: X
  labels_key: y
  flatten: true
  normalization: unit
```

This block loads all samples in the file. The current NPZ loader does not select the optional `split` array; create a train-only file if the protocol requires it. Use `flatten: false` for DeepCluster's NCHW input. For CIFAR-100-20, prepare coarse labels from the original CIFAR-100 files, set `labels_key: y_coarse` and `parameters.n_clusters: 20` in the full configuration.

Match preprocessing to the method: USPS source features can be negative, so scaling by their maximum absolute value does not put them in `[0, 1]` for Bernoulli reconstruction. Keep Reuters preprocessing consistent with the pinned feature recipe. Record source versions, splits, preprocessing and checksums with experiment outputs; see the [benchmark guide](../docs/benchmarking.md).
