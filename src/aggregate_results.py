"""Aggregate the sweep, apply the pre-registered verdict rules, emit tables and figures.

Reads  results/runs.jsonl, results/curves.jsonl, claims.yaml
Writes results/table_by_config.csv, results/summary.json, results/VERDICTS.md,
       figures/*.png
"""
import json, os, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ADAPTIVE = ["adagrad", "rmsprop", "adam"]
NON_ADAPTIVE = ["sgd", "heavy_ball"]
ORDER = NON_ADAPTIVE + ADAPTIVE
PAPER_BEST_LR = {"sgd": 0.5, "heavy_ball": 0.5, "adagrad": 0.01, "rmsprop": 0.0003, "adam": 0.0003}
DEFAULT_LR = {"adagrad": 0.01, "rmsprop": 0.001, "adam": 0.001}
PAPER = dict(nonadaptive=("sgd", 7.65, 0.14), adaptive=("rmsprop", 9.60, 0.19), gap=1.95)
COLORS = {"sgd": "#1f77b4", "heavy_ball": "#4c9be8", "adagrad": "#d62728",
          "rmsprop": "#ff7f0e", "adam": "#9467bd"}


def load():
    rows = [json.loads(l) for l in open("results/runs.jsonl")]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "config"} for r in rows])
    return df


def per_config(df):
    g = df.groupby(["optimizer", "lr"])
    t = g.agg(
        n_seeds=("seed", "count"),
        n_diverged=("diverged", "sum"),
        train_loss_mean=("final_train_loss", "mean"),
        train_loss_std=("final_train_loss", "std"),
        train_err_mean=("final_train_err", "mean"),
        test_err_mean=("final_test_err", "mean"),
        test_err_std=("final_test_err", "std"),
        test_err_min=("final_test_err", "min"),
        test_err_max=("final_test_err", "max"),
        best_test_err_mean=("best_test_err", "mean"),
        wall_s=("wall_seconds", "mean"),
    ).reset_index()
    t["finite"] = np.isfinite(t["train_loss_mean"]) & (t["n_diverged"] == 0)
    return t


def select_lr(t):
    """The paper's rule: lowest TRAINING loss at the end of the fixed epoch budget."""
    sel = {}
    for opt in ORDER:
        sub = t[(t.optimizer == opt) & t.finite]
        if len(sub) == 0:
            sel[opt] = None
            continue
        sel[opt] = float(sub.loc[sub.train_loss_mean.idxmin(), "lr"])
    return sel


def main():
    os.makedirs("figures", exist_ok=True)
    df = load()
    t = per_config(df)
    t.sort_values(["optimizer", "lr"]).to_csv("results/table_by_config.csv", index=False)
    sel = select_lr(t)

    def at(opt):
        r = t[(t.optimizer == opt) & (t.lr == sel[opt])].iloc[0]
        return r

    rows = {o: at(o) for o in ORDER if sel[o] is not None}
    S = {"selected_lr": sel,
         "at_selected": {o: dict(lr=sel[o],
                                 test_err_mean=float(r.test_err_mean),
                                 test_err_std=float(r.test_err_std),
                                 test_err_min=float(r.test_err_min),
                                 test_err_max=float(r.test_err_max),
                                 train_loss_mean=float(r.train_loss_mean),
                                 train_err_mean=float(r.train_err_mean))
                         for o, r in rows.items()}}

    # ---------------------------------------------------------------- C1
    na = min(NON_ADAPTIVE, key=lambda o: rows[o].test_err_mean)
    ad = min(ADAPTIVE, key=lambda o: rows[o].test_err_mean)
    gap = float(rows[ad].test_err_mean - rows[na].test_err_mean)
    pooled = float(np.sqrt((rows[na].test_err_std ** 2 + rows[ad].test_err_std ** 2) / 2))
    c1_ok = bool(gap > 0 and gap > 2 * pooled)
    S["C1"] = dict(best_nonadaptive=na, best_nonadaptive_err=float(rows[na].test_err_mean),
                   best_nonadaptive_std=float(rows[na].test_err_std),
                   best_adaptive=ad, best_adaptive_err=float(rows[ad].test_err_mean),
                   best_adaptive_std=float(rows[ad].test_err_std),
                   gap_pct_points=gap, pooled_std=pooled, gap_over_pooled_std=gap / pooled,
                   paper_gap=PAPER["gap"],
                   verdict="REPRODUCED" if c1_ok else "DIVERGED")

    # ------------------------------------------------- C1b: the most generous alternative
    # If the LR had been selected by TEST error instead (which the paper does not do, and
    # which leaks the test set), would the claim hold? Reported so that the divergence
    # cannot be blamed on the selection rule.
    best_by_test = {o: t[(t.optimizer == o) & t.finite].test_err_mean.min() for o in ORDER}
    na_b = min(NON_ADAPTIVE, key=lambda o: best_by_test[o])
    ad_b = min(ADAPTIVE, key=lambda o: best_by_test[o])
    gap_b = float(best_by_test[ad_b] - best_by_test[na_b])
    S["C1b_test_selected"] = dict(
        note="oracle selection by test error; not the paper's rule",
        best_nonadaptive=na_b, err=float(best_by_test[na_b]),
        best_adaptive=ad_b, adaptive_err=float(best_by_test[ad_b]),
        gap_pct_points=gap_b, gap_over_pooled_std=gap_b / pooled,
        verdict="REPRODUCED" if gap_b > 2 * pooled else "DIVERGED")

    # ---------------------------------------------------------------- C2
    min_ad_tl = float(min(rows[o].train_loss_mean for o in ADAPTIVE))
    min_na_tl = float(min(rows[o].train_loss_mean for o in NON_ADAPTIVE))
    S["C2"] = dict(min_adaptive_train_loss=min_ad_tl, min_nonadaptive_train_loss=min_na_tl,
                   diff=min_ad_tl - min_na_tl,
                   verdict="REPRODUCED" if min_ad_tl <= min_na_tl + 0.01 else "DIVERGED")

    # ---------------------------------------------------------------- C3
    na_lr = max(sel[o] for o in NON_ADAPTIVE)
    ad_lr = min(sel[o] for o in ("adam", "rmsprop"))
    ratio = na_lr / ad_lr
    S["C3"] = dict(nonadaptive_lr=na_lr, adam_rmsprop_lr=ad_lr, ratio=float(ratio),
                   paper_ratio=0.5 / 0.0003,
                   exact_match={o: dict(ours=sel[o], paper=PAPER_BEST_LR[o],
                                        match=bool(sel[o] == PAPER_BEST_LR[o])) for o in ORDER},
                   n_exact_matches=int(sum(sel[o] == PAPER_BEST_LR[o] for o in ORDER)),
                   verdict="REPRODUCED" if ratio >= 100 else "DIVERGED")

    # ---------------------------------------------------------------- C4
    dflt = {o: dict(default=DEFAULT_LR[o], selected=sel[o], default_selected=bool(sel[o] == DEFAULT_LR[o]))
            for o in ADAPTIVE}
    S["C4"] = dict(table=dflt,
                   verdict="REPRODUCED" if not dflt["adam"]["default_selected"] else "DIVERGED")

    # ---------------------------------------------------------------- C5 (obsession marker)
    na_seeds = df[(df.optimizer == na) & (df.lr == sel[na])].sort_values("seed").final_test_err.values
    ad_seeds = df[(df.optimizer == ad) & (df.lr == sel[ad])].sort_values("seed").final_test_err.values
    pairs = list(itertools.product(na_seeds, ad_seeds))
    win_rate = float(np.mean([a < b for a, b in pairs]))
    stds = {o: float(rows[o].test_err_std) for o in ORDER}
    all_std = t[t.finite].test_err_std
    S["C5"] = dict(seed_std_at_selected=stds,
                   paper_std={"sgd": 0.14, "rmsprop": 0.19},
                   our_median_std_all_configs=float(all_std.median()),
                   our_max_std_all_configs=float(all_std.max()),
                   seed_level_win_rate=win_rate, n_pairs=len(pairs),
                   nonadaptive_seed_errors=[float(x) for x in na_seeds],
                   adaptive_seed_errors=[float(x) for x in ad_seeds],
                   gap_over_pooled_std=gap / pooled,
                   seed_range_na=float(na_seeds.max() - na_seeds.min()),
                   seed_range_ad=float(ad_seeds.max() - ad_seeds.min()))

    # ---------------------------------------------------------------- C6
    within = {}
    for o in ORDER:
        sub = t[(t.optimizer == o) & t.finite]
        within[o] = float(sub.test_err_mean.max() - sub.test_err_mean.min())
    between = float(max(r.test_err_mean for r in rows.values()) - min(r.test_err_mean for r in rows.values()))
    med_within = float(np.median(list(within.values())))
    S["C6"] = dict(within_optimizer_lr_spread=within, median_within=med_within,
                   between_optimizer_spread_at_tuned_lr=between,
                   n_diverged_runs=int(df.diverged.sum()),
                   diverged_configs=[f"{r.optimizer}@{r.lr}" for _, r in t[t.n_diverged > 0].iterrows()],
                   verdict="REPRODUCED" if med_within > between else "DIVERGED")

    S["run_counts"] = dict(total_runs=int(len(df)), diverged=int(df.diverged.sum()),
                           total_wall_minutes=float(df.wall_seconds.sum() / 60))
    json.dump(S, open("results/summary.json", "w"), indent=2)

    # ---------------------------------------------------------------- figures
    # 1. test error vs lr, per optimizer, with seed spread
    fig, ax = plt.subplots(figsize=(8, 5))
    for o in ORDER:
        sub = t[(t.optimizer == o) & t.finite].sort_values("lr")
        ax.errorbar(sub.lr, sub.test_err_mean, yerr=sub.test_err_std, marker="o", capsize=3,
                    label=o, color=COLORS[o])
        d = t[(t.optimizer == o) & (~t.finite)]
        if len(d):
            ax.scatter(d.lr, [ax.get_ylim()[1]] * len(d), marker="x", color=COLORS[o])
    ax.set_xscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("test error (%)")
    ax.set_title("Learning-rate sensitivity per optimizer (mean ± 1 sd over 5 seeds)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/fig1_lr_sensitivity.png", dpi=150); plt.close(fig)

    # 2. seed spread at the selected setting
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, o in enumerate(ORDER):
        e = df[(df.optimizer == o) & (df.lr == sel[o])].final_test_err.values
        ax.scatter([i] * len(e), e, color=COLORS[o], zorder=3, s=45)
        ax.plot([i - .22, i + .22], [e.mean()] * 2, color="k", lw=2, zorder=4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{o}\nlr={sel[o]:g}" for o in ORDER], fontsize=9)
    ax.set_ylabel("test error (%)")
    ax.set_title("Every seed at each optimizer's tuned LR (bar = mean)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig("figures/fig2_seed_spread.png", dpi=150); plt.close(fig)

    # 3. training loss vs test error at selected settings (the paper's core picture)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for o in ORDER:
        r = rows[o]
        ax.scatter(r.train_loss_mean, r.test_err_mean, color=COLORS[o], s=70,
                   marker="o" if o in NON_ADAPTIVE else "^")
        ax.annotate(f" {o}", (r.train_loss_mean, r.test_err_mean), fontsize=9)
    ax.set_xlabel("final training loss"); ax.set_ylabel("test error (%)")
    ax.set_title("Same training loss, different test error?\n(circles = non-adaptive, triangles = adaptive)")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/fig3_trainloss_vs_testerr.png", dpi=150); plt.close(fig)

    # 4. curves at selected settings
    try:
        curves = [json.loads(l) for l in open("results/curves.jsonl")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for o in ORDER:
            cs = [c for c in curves if c["optimizer"] == o and c["lr"] == sel[o]]
            if not cs:
                continue
            tr = np.array([[p["train_loss"] for p in c["curve"]] for c in cs])
            te = np.array([[p["test_err"] for p in c["curve"]] for c in cs])
            ep = np.arange(1, tr.shape[1] + 1)
            axes[0].plot(ep, tr.mean(0), color=COLORS[o], label=o)
            axes[0].fill_between(ep, tr.min(0), tr.max(0), color=COLORS[o], alpha=.15)
            axes[1].plot(ep, te.mean(0), color=COLORS[o], label=o)
            axes[1].fill_between(ep, te.min(0), te.max(0), color=COLORS[o], alpha=.15)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss"); axes[0].set_yscale("log")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test error (%)")
        for a in axes:
            a.grid(alpha=.3); a.legend(fontsize=8)
        axes[0].set_title("Training loss at tuned LR (band = seed min/max)")
        axes[1].set_title("Test error at tuned LR (band = seed min/max)")
        fig.tight_layout(); fig.savefig("figures/fig4_curves.png", dpi=150); plt.close(fig)
    except FileNotFoundError:
        pass

    # ---------------------------------------------------------------- verdict sheet
    L = []
    W = L.append
    W("# Verdicts against the pre-registered claims\n")
    W(f"Runs: {S['run_counts']['total_runs']} "
      f"({S['run_counts']['diverged']} diverged), "
      f"{S['run_counts']['total_wall_minutes']:.0f} CPU-minutes total.\n")
    W("| claim | verdict |")
    W("|---|---|")
    for c in ["C1", "C2", "C3", "C4", "C6"]:
        W(f"| {c} | **{S[c]['verdict']}** |")
    W("| C5 (variance, report-only) | reported |\n")
    W("## Selected learning rates (paper's rule: lowest final training loss)\n")
    W("| optimizer | our LR | paper LR | match | test err % (mean ± sd over 5 seeds) | final train loss |")
    W("|---|---|---|---|---|---|")
    for o in ORDER:
        r = rows[o]
        W(f"| {o} | {sel[o]:g} | {PAPER_BEST_LR[o]:g} | "
          f"{'yes' if sel[o] == PAPER_BEST_LR[o] else 'no'} | "
          f"{r.test_err_mean:.2f} ± {r.test_err_std:.2f} | {r.train_loss_mean:.4f} |")
    open("results/VERDICTS.md", "w").write("\n".join(L) + "\n")
    print(json.dumps({k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else "")
                      for k, v in S.items() if k.startswith("C")}, indent=2))
    print("wrote results/summary.json, results/VERDICTS.md, results/table_by_config.csv, figures/*.png")


if __name__ == "__main__":
    main()
