"""Run the full pre-registered sweep: 5 optimizers x the paper's LR grid x 5 seeds.

Writes one JSON object per run to results/runs.jsonl (append-only, resumable) and the
per-epoch curves to results/curves.jsonl.

    python src/sweep.py --workers 2 --seeds 5
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))

RUNS = "results/runs.jsonl"
CURVES = "results/curves.jsonl"


def job(args):
    opt, lr, seed, cfg_kwargs = args
    from train_harness import run_one, Config          # imported inside the worker
    rec = run_one(opt, lr, seed, Config(**cfg_kwargs))
    curve = rec.pop("curve")
    return rec, {"optimizer": opt, "lr": lr, "seed": seed, "curve": curve}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--data", default="data/cifar10.npz")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n_train", type=int, default=None)
    a = p.parse_args()

    from train_harness import LR_GRIDS, Config
    cfg_kwargs = {"data_path": a.data}
    if a.epochs:
        cfg_kwargs["epochs"] = a.epochs
    if a.n_train:
        cfg_kwargs["n_train"] = a.n_train

    os.makedirs("results", exist_ok=True)
    done = set()
    if os.path.exists(RUNS):
        for line in open(RUNS):
            try:
                r = json.loads(line)
                done.add((r["optimizer"], r["lr"], r["seed"]))
            except Exception:
                pass

    jobs = [(o, lr, s, cfg_kwargs)
            for o, grid in LR_GRIDS.items()
            for lr in grid
            for s in range(a.seeds)
            if (o, lr, s) not in done]
    total = sum(len(g) for g in LR_GRIDS.values()) * a.seeds
    print(f"{len(jobs)} runs to do ({total} total, {len(done)} already complete)", flush=True)

    t0 = time.time()
    with open(RUNS, "a") as fr, open(CURVES, "a") as fc, Pool(a.workers) as pool:
        for i, (rec, cur) in enumerate(pool.imap_unordered(job, jobs), 1):
            fr.write(json.dumps(rec) + "\n"); fr.flush()
            fc.write(json.dumps(cur) + "\n"); fc.flush()
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] {rec['optimizer']:<11} lr={rec['lr']:<8} seed={rec['seed']} "
                  f"train_loss={rec['final_train_loss']:.4f} test_err={rec['final_test_err']:.2f} "
                  f"{'DIVERGED ' if rec['diverged'] else ''}"
                  f"| {el/60:.1f}m elapsed, ~{el/i*(len(jobs)-i)/60:.0f}m left", flush=True)
    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
