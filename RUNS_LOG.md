# Runs log — every decision in the order it was made

Kept so that anyone reading the results can see what was decided before the data existed,
what was decided because of a pilot, and what went wrong on the way.

---

## P0 — Pre-registration

1. **Target chosen:** Wilson et al. (2017), *The Marginal Value of Adaptive Gradient Methods in
   Machine Learning*, NeurIPS 2017 (arXiv:1705.08292). Chosen over Schmidt et al. (2021)
   "Crowded Valley" and Reddi et al. (2018) AMSGrad because it publishes, for CIFAR-10:
   the complete LR grid per optimizer (Appendix D), the selected LR per optimizer, the headline
   numbers (SGD 7.65 ± 0.14%, RMSProp 9.60 ± 0.19%), the LR-selection rule, and its own
   seed count and standard deviation. The last item is what makes the variance comparison possible.
2. **Claims C1–C6 written with verdict rules, `claims.yaml` frozen.** Nothing had been run.
3. **Under-specification recorded.** The paper does not state, for CIFAR-10: batch size,
   decay factor δ, decay frequency k, weight decay, ε, or augmentation. Every one of these is a
   guess on our side and is listed in deviations D5/D6 so that a divergence traceable to them is
   falsifiable rather than hand-wavy.

## P1 — Harness, and the three things that went wrong

4. **Data.** `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` is blocked by the sandbox's
   egress policy (403 at the proxy). Fell back to a GitHub mirror storing CIFAR-10 as per-class
   JPEGs — 50,000 train / 10,000 test, correct class counts. Logged as **D4**;
   `src/prepare_data.py` tries the canonical source first and only then the mirror.
5. **Framework.** PyTorch is unavailable here (`download.pytorch.org` also blocked). Used
   JAX + optax + flax, all installable from PyPI. optax provides the five optimizers with
   standard implementations: `sgd`, `sgd(momentum=0.9)`, `adagrad`, `rmsprop(decay=0.99)`, `adam`.
6. **Pilot 1 — too slow.** 4-conv net, 32×32 inputs, 10k images: 13 min/run → 30 h for 140 runs.
   Benchmarked alternatives; 16×16 average-pooled inputs gave a 5× speedup at 2.1 s/epoch (**D7**).
7. **Bug found in our own harness (not the paper's).** Per-epoch curve evaluation used the first
   *N* test images as a subsample, but the mirror stores images class by class, so the "test error"
   in the curve was computed on two classes only. Fixed by shuffling the test set once with a fixed
   permutation. Any result produced before this fix was discarded.
8. **Bug 2 — decay schedule.** `decay_every` was a constant 3 epochs regardless of the epoch budget,
   so a 30-epoch run decayed the LR 9 times (×1/512) and effectively stopped training a third of the
   way in. Changed to `k = epochs // 5`, i.e. 4 decays per run at any budget.
9. **Pilot 2 — the regime problem.** With dropout (0.25 per block, 0.5 head) the scaled-down net
   could not push training error below ~45% within any affordable budget
   (10k images / 20 epochs: train err 47%; 2k images / 40 epochs: train err 44.8%).
   Wilson et al.'s claim is explicitly about methods that reach *the same training loss* and then
   differ on test error — with the net underfitting, C1 and C2 are not testable at all.
   Pilot with dropout removed: training error **0.0%**, training loss 0.023, test error 50.4%.
   **Decision: dropout disabled, logged as D8**, with the reasoning above, before the sweep ran.
10. **Final config locked:** 2,000-image training subset (fixed across all runs), full 10,000-image
    test set, 16×16 inputs, 3 conv blocks (16/32/64) + BN + 128-unit head, no dropout,
    batch 64, 30 epochs, fixed-decay ×0.5 every 6 epochs, random horizontal flips, weight decay 0.
    Seeds 0–4 vary initialisation, batch order and augmentation only.

## P2 — Sweep

11. 28 configurations (the paper's exact grids) × 5 seeds = **140 runs**, 2 worker processes,
    append-only `results/runs.jsonl` + `results/curves.jsonl`, resumable.
12. Runs that produced non-finite training loss were recorded with `diverged: true` and kept in the
    results file. They are excluded from LR selection (a diverged run has no valid training loss)
    and reported separately — never silently dropped.

## P3 — Analysis

13. LR selected per optimizer by the paper's rule: **lowest final training loss**, averaged over
    seeds, among non-diverged configurations. Test error is read off at that LR and never used to
    choose it.
14. Verdicts computed mechanically from the frozen rules in `claims.yaml` by
    `src/aggregate_results.py`. No claim was edited after results existed.
