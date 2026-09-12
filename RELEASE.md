# Releasing this artifact (GitHub + Zenodo DOI)

Run these from inside the unzipped folder. Git history (5 commits, P0 → P4) is already in place —
do **not** re-init the repo, or the pre-registration-before-results ordering is lost.

## 1. GitHub

```bash
git log --oneline            # should show P0 at the bottom, P4 at the top
gh repo create a018-lr-sensitivity-reproduction --public --source=. --remote=origin --push
# or, without the gh CLI:
#   create an empty repo on github.com, then:
#   git remote add origin https://github.com/<you>/a018-lr-sensitivity-reproduction.git
#   git branch -M main && git push -u origin main
```

`data/` is gitignored (163 MB). Anyone reproducing runs `python src/prepare_data.py` to rebuild it.

## 2. Zenodo DOI

1. Sign in to <https://zenodo.org> with GitHub.
2. **Settings → GitHub**, find the repository, toggle it **On**. (The toggle must be flipped
   *before* the release — Zenodo only sees releases created after the webhook exists.)
3. Back in the repo:

```bash
git tag -a v1.0.0 -m "A.01.8: reduced-scale reproduction of Wilson et al. (2017)"
git push origin v1.0.0
gh release create v1.0.0 --title "v1.0.0" --notes "Pre-registered reduced-scale reproduction. C1 diverged (gap inverts inside seed noise); C2, C3, C4, C6 reproduced; seed variance reported."
```

4. Zenodo picks up the release within a minute or two and mints the DOI.
5. Copy the DOI into two places and push the change:
   - a `doi:` line in `CITATION.cff`
   - a DOI badge at the top of `README.md`:
     `[![DOI](https://zenodo.org/badge/DOI/<your-doi>.svg)](https://doi.org/<your-doi>)`

## 3. Optional — the short note

The divergence is arguably worth a paragraph at a reproducibility venue: the paper's headline gap
inverts and falls inside seed noise at reduced scale, and the published learning-rate grids do not
bracket the adaptive optima for a different architecture. Candidates: the ML Reproducibility
Challenge, a ReScience C submission, or a workshop reproducibility track. Lead with the variance
finding, not the sign flip — the sign flip is the attention-getter, the variance is the argument.

## Checklist

- [ ] `git log` shows P0 before P2
- [ ] pushed to GitHub, repo public
- [ ] Zenodo GitHub toggle **on** before tagging
- [ ] release `v1.0.0` created, DOI minted
- [ ] DOI added to `CITATION.cff` and the README badge, pushed
