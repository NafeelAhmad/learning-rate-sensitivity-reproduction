"""Single training run: one optimizer, one learning rate, one seed.

Faithful-where-possible reproduction of the CIFAR-10 experiment in
Wilson et al. (2017), arXiv:1705.08292. Deviations are declared in claims.yaml.
"""
from __future__ import annotations
import functools, json, time
from dataclasses import dataclass, asdict

import numpy as np
import jax, jax.numpy as jnp
import optax
import flax.linen as nn

# ----------------------------------------------------------------------------- config
DEFAULT_DATA = "data/cifar10.npz"

# LR grids transcribed from Wilson et al. (2017), Appendix D, CIFAR-10 column.
LR_GRIDS = {
    "sgd":        [2.0, 1.0, 0.5, 0.25, 0.05, 0.01],
    "heavy_ball": [2.0, 1.0, 0.5, 0.25, 0.05, 0.01],
    "adagrad":    [0.1, 0.05, 0.01, 0.0075, 0.005],
    "rmsprop":    [0.005, 0.001, 0.0005, 0.0003, 0.0001],
    "adam":       [0.005, 0.001, 0.0005, 0.0003, 0.0001, 0.00005],
}
ADAPTIVE = {"adagrad", "rmsprop", "adam"}
NON_ADAPTIVE = {"sgd", "heavy_ball"}
LIBRARY_DEFAULT_LR = {"adagrad": 0.01, "rmsprop": 0.001, "adam": 0.001}


@dataclass
class Config:
    n_train: int = 2000           # D2
    img_size: int = 16            # D7: 32x32 -> 16x16 by 2x2 average pooling
    epochs: int = 30              # D3
    batch_size: int = 64          # D6
    decay_factor: float = 0.5     # fixed-decay, delta
    decay_every: int = 0          # fixed-decay, k (epochs); 0 => epochs//5 (4 decays per run)
    momentum: float = 0.9         # heavy ball
    rmsprop_decay: float = 0.99
    eps: float = 1e-8
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    weight_decay: float = 0.0     # D5
    block_dropout: float = 0.0    # D8
    head_dropout: float = 0.0     # D8
    hflip: bool = True            # D5
    curve_eval_n: int = 1000      # per-epoch curve eval subsample; final epoch uses the full sets
    subset_seed: int = 12345      # fixed across ALL runs; seeds vary init/order/dropout only
    data_path: str = DEFAULT_DATA


# ----------------------------------------------------------------------------- data
_CACHE: dict = {}

def load_data(cfg: Config):
    key = (cfg.data_path, cfg.n_train, cfg.subset_seed, cfg.img_size)
    if key in _CACHE:
        return _CACHE[key]
    d = np.load(cfg.data_path)
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    rng = np.random.default_rng(cfg.subset_seed)
    idx = rng.permutation(len(Xtr))[: cfg.n_train]
    Xtr, ytr = Xtr[idx], ytr[idx]
    # The mirror stores images class by class. Shuffle the test set once with a fixed
    # permutation so that the per-epoch curve subsample is class-balanced.
    tidx = np.random.default_rng(cfg.subset_seed + 1).permutation(len(Xte))
    Xte, yte = Xte[tidx], yte[tidx]
    mean = Xtr.reshape(-1, 3).mean(0) / 255.0
    std = Xtr.reshape(-1, 3).std(0) / 255.0
    def prep(X):
        X = (X.astype(np.float32) / 255.0 - mean) / std
        if cfg.img_size != 32:                      # D7: average-pool downsample
            f = 32 // cfg.img_size
            X = X.reshape(len(X), cfg.img_size, f, cfg.img_size, f, 3).mean((2, 4))
        return np.ascontiguousarray(X, dtype=np.float32)
    out = (prep(Xtr), ytr.astype(np.int32), prep(Xte), yte.astype(np.int32))
    _CACHE[key] = out
    return out


# ----------------------------------------------------------------------------- model
class SmallVGG(nn.Module):
    """VGG-style conv net with BN + dropout, scaled down for CPU (deviation D1).

    Structure mirrors the cifar.torch VGG family -- 3x3 convs, BN, ReLU, 2x2 max-pool,
    dropout between blocks, one BN+dropout dense head -- at 3 conv layers and
    16/32/64 channels instead of 16 conv layers and 64-512 channels.
    """
    blocks: tuple = (16, 32, 64)
    head: int = 128
    block_dropout: float = 0.1
    head_dropout: float = 0.3

    @nn.compact
    def __call__(self, x, train: bool):
        for feats in self.blocks:
            x = nn.Conv(feats, (3, 3), padding="SAME", use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = nn.Dropout(self.block_dropout, deterministic=not train)(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.head, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.head_dropout, deterministic=not train)(x)
        return nn.Dense(10)(x)


# ----------------------------------------------------------------------------- optimizers
def make_optimizer(name: str, lr_schedule, cfg: Config):
    if name == "sgd":
        return optax.sgd(lr_schedule)
    if name == "heavy_ball":
        return optax.sgd(lr_schedule, momentum=cfg.momentum, nesterov=False)
    if name == "adagrad":
        return optax.adagrad(lr_schedule, eps=1e-10)
    if name == "rmsprop":
        return optax.rmsprop(lr_schedule, decay=cfg.rmsprop_decay, eps=cfg.eps)
    if name == "adam":
        return optax.adam(lr_schedule, b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.eps)
    raise ValueError(name)


# ----------------------------------------------------------------------------- train
def run_one(optimizer: str, lr: float, seed: int, cfg: Config, keep_curve: bool = True) -> dict:
    Xtr, ytr, Xte, yte = load_data(cfg)
    n = len(Xtr)
    steps_per_epoch = n // cfg.batch_size

    model = SmallVGG(block_dropout=cfg.block_dropout, head_dropout=cfg.head_dropout)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    variables = model.init(init_key, jnp.zeros((1, cfg.img_size, cfg.img_size, 3), jnp.float32), train=False)
    params, bstats = variables["params"], variables["batch_stats"]

    # fixed-decay: multiply by decay_factor every decay_every epochs
    k = cfg.decay_every or max(1, cfg.epochs // 5)
    sched = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales={(e * steps_per_epoch): cfg.decay_factor
                               for e in range(k, cfg.epochs, k)},
    )
    tx = make_optimizer(optimizer, sched, cfg)
    opt_state = tx.init(params)

    def loss_fn(params, bstats, xb, yb, dkey):
        logits, new_state = model.apply(
            {"params": params, "batch_stats": bstats}, xb, train=True,
            mutable=["batch_stats"], rngs={"dropout": dkey})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        return loss, new_state["batch_stats"]

    @jax.jit
    def train_step(params, bstats, opt_state, xb, yb, dkey):
        (loss, bstats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, bstats, xb, yb, dkey)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, bstats, opt_state, loss

    @jax.jit
    def eval_batch(params, bstats, xb, yb):
        logits = model.apply({"params": params, "batch_stats": bstats}, xb, train=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb).sum()
        correct = (logits.argmax(-1) == yb).sum()
        return loss, correct

    def evaluate(params, bstats, X, y, bs=2000, limit=None):
        if limit is not None and limit < len(X):
            X, y = X[:limit], y[:limit]
        L = C = 0.0
        for i in range(0, len(X), bs):
            l, c = eval_batch(params, bstats, X[i:i + bs], y[i:i + bs])
            L += float(l); C += float(c)
        return L / len(X), 100.0 * (1.0 - C / len(X))   # (mean loss, error %)

    rng_np = np.random.default_rng(seed)
    curve, diverged, t0 = [], False, time.time()

    for epoch in range(cfg.epochs):
        order = rng_np.permutation(n)
        for s in range(steps_per_epoch):
            bidx = order[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            xb = Xtr[bidx]
            if cfg.hflip:
                flip = rng_np.random(len(bidx)) < 0.5
                xb = np.where(flip[:, None, None, None], xb[:, :, ::-1, :], xb)
            key, dkey = jax.random.split(key)
            params, bstats, opt_state, loss = train_step(
                params, bstats, opt_state, jnp.asarray(xb), jnp.asarray(ytr[bidx]), dkey)
        last = (epoch == cfg.epochs - 1)
        lim = None if last else cfg.curve_eval_n   # full eval only on the final epoch
        tr_loss, tr_err = evaluate(params, bstats, Xtr, ytr, limit=lim)
        te_loss, te_err = evaluate(params, bstats, Xte, yte, limit=lim)
        if not np.isfinite(tr_loss):
            diverged = True
        if keep_curve:
            curve.append(dict(epoch=epoch + 1, train_loss=tr_loss, train_err=tr_err,
                              test_loss=te_loss, test_err=te_err))
        if diverged:
            break

    finite = [c for c in curve if np.isfinite(c["train_loss"])]
    rec = dict(
        optimizer=optimizer, lr=lr, seed=seed, diverged=bool(diverged),
        epochs_completed=len(curve),
        final_train_loss=curve[-1]["train_loss"] if curve else float("nan"),
        final_train_err=curve[-1]["train_err"] if curve else float("nan"),
        final_test_err=curve[-1]["test_err"] if curve else float("nan"),
        final_test_loss=curve[-1]["test_loss"] if curve else float("nan"),
        best_test_err=min([c["test_err"] for c in finite], default=float("nan")),
        min_train_loss=min([c["train_loss"] for c in finite], default=float("nan")),
        wall_seconds=round(time.time() - t0, 2),
        config=asdict(cfg),
        curve=curve if keep_curve else None,
    )
    return rec


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--n_train", type=int, default=Config.n_train)
    p.add_argument("--bs", type=int, default=Config.batch_size)
    p.add_argument("--block_dropout", type=float, default=Config.block_dropout)
    p.add_argument("--head_dropout", type=float, default=Config.head_dropout)
    p.add_argument("--data", default=DEFAULT_DATA)
    a = p.parse_args()
    cfg = Config(epochs=a.epochs, n_train=a.n_train, batch_size=a.bs,
                 block_dropout=a.block_dropout, head_dropout=a.head_dropout, data_path=a.data)
    r = run_one(a.optimizer, a.lr, a.seed, cfg)
    r.pop("curve")
    print(json.dumps(r, indent=2))
