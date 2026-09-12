# Verdicts against the pre-registered claims

Runs: 140 (0 diverged), 63 CPU-minutes total.

| claim | verdict |
|---|---|
| C1 | **DIVERGED** |
| C2 | **REPRODUCED** |
| C3 | **REPRODUCED** |
| C4 | **REPRODUCED** |
| C6 | **REPRODUCED** |
| C5 (variance, report-only) | reported |

## Selected learning rates (paper's rule: lowest final training loss)

| optimizer | our LR | paper LR | match | test err % (mean ± sd over 5 seeds) | final train loss |
|---|---|---|---|---|---|
| sgd | 1 | 0.5 | no | 50.36 ± 0.69 | 0.0637 |
| heavy_ball | 0.05 | 0.5 | no | 51.04 ± 0.51 | 0.0722 |
| adagrad | 0.1 | 0.01 | no | 50.69 ± 0.63 | 0.1271 |
| rmsprop | 0.005 | 0.0003 | no | 50.01 ± 0.69 | 0.0807 |
| adam | 0.005 | 0.0003 | no | 51.17 ± 0.55 | 0.0381 |
