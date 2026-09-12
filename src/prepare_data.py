"""Fetch CIFAR-10 and cache it as a single .npz.

The canonical release (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) is
preferred. If that host is unreachable (it is blocked by the sandbox egress policy
this artifact was produced in), fall back to the JPEG mirror -- see deviation D4 in
claims.yaml.

Usage:  python src/prepare_data.py --out data/cifar10.npz
"""
import argparse, io, os, sys, tarfile, pickle, urllib.request
import numpy as np

CANON_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MIRROR_REPO = "https://github.com/YoongiKim/CIFAR-10-images.git"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def from_canonical(tmp):
    path = os.path.join(tmp, "cifar-10-python.tar.gz")
    urllib.request.urlretrieve(CANON_URL, path)
    Xtr, ytr = [], []
    with tarfile.open(path) as tf:
        for i in range(1, 6):
            d = pickle.load(tf.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="bytes")
            Xtr.append(d[b"data"]); ytr += d[b"labels"]
        dt = pickle.load(tf.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
    rs = lambda a: np.concatenate(a).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (rs(Xtr), np.array(ytr, np.int64),
            rs([dt[b"data"]]), np.array(dt[b"labels"], np.int64))


def from_mirror(tmp):
    from PIL import Image
    root = os.path.join(tmp, "CIFAR-10-images")
    if not os.path.isdir(root):
        os.system(f"git clone --depth 1 --quiet {MIRROR_REPO} {root}")
    out = []
    for split in ("train", "test"):
        X, y = [], []
        for ci, c in enumerate(CLASSES):
            d = os.path.join(root, split, c)
            for fn in sorted(os.listdir(d)):
                X.append(np.asarray(Image.open(os.path.join(d, fn)).convert("RGB"), np.uint8))
                y.append(ci)
        out += [np.stack(X), np.array(y, np.int64)]
    return tuple(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/cifar10.npz")
    p.add_argument("--tmp", default="/tmp/cifar_src")
    a = p.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    try:
        Xtr, ytr, Xte, yte = from_canonical(a.tmp)
        src = "canonical"
    except Exception as e:
        print(f"canonical source failed ({e}); falling back to JPEG mirror (deviation D4)", file=sys.stderr)
        Xtr, ytr, Xte, yte = from_mirror(a.tmp)
        src = "jpeg_mirror"
    np.savez_compressed(a.out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, source=np.array(src))
    print(f"wrote {a.out}  source={src}  train={Xtr.shape} test={Xte.shape}")
