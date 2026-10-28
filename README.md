# SGKR & SPHR — A Recommender System for Electronics Products

**English** · [Tiếng Việt](README.vi.md)

An individual *Advanced Machine Learning* major assignment. I propose two top-N recommenders, **SGKR** (a closed-form
item–item model) and **SPHR** (a history-based model trained by gradient descent), and compare them with five
published baselines on Amazon Electronics. Every model shares the same data, split, full-catalog evaluator and
metrics, and every reported number can be re-derived from the stored run files.

## Highlights

- **SGKR has the best NDCG@20 at full density (0.0383)** and beats LightGCN, XSimGCL and DirectAU with
  statistical significance (Holm-adjusted p = 0.0006), while training in **2.5 minutes on a CPU**, about 17×
  faster than the fastest graph neural network.
- **SPHR is the strongest model on sparse data:** best NDCG@20 at 25 % and 50 % of the training interactions.
- **Against closed-form models of its own kind**, SGKR is statistically tied with full EASE at full density,
  which needs a 7.5 GB matrix that cannot be stored, and beats EASE pruned to a storable size and GF-CF.
- **Limits:** SGKR is not the best at 50 % density, and all results come from a single seed on the validation
  split (see [Limitations](#limitations)).

## Contents

[Reproducing the results](#reproducing-the-results) · [Demo](#demo) · [Models](#models) · [Data](#data) ·
[Results](#results) · [Evaluation protocol](#evaluation-protocol) · [Limitations](#limitations) ·
[Repository layout](#repository-layout) · [References](#references)

## Reproducing the results

```bash
conda create -n AML python=3.11 && conda activate AML
pip install -r requirements.txt

python scripts/prepare_data.py                         # download from SNAP and build the dataset
python scripts/audit_data.py                           # data checks

python scripts/train_model.py --model sgkr --density 1.0
#   --model    sgkr | sphr | lightgcn | xsimgcl | directau | ease | gfcf
#   --density  1.0 | 0.75 | 0.5 | 0.25

python scripts/summarize_results.py                    # results/summary.csv
python scripts/make_report.py                          # main figures H1–H8, tables B1–B4
python scripts/make_closed_form_report.py              # closed-form figures, tables B5–B6
python scripts/verify_results.py                       # re-derive every metric from the stored files
python -m pytest -q tests
```

Each run writes `results/<model>/d<density>/seed42/` with `metrics.json`, the validation predictions, the
checkpoint, `history.csv` for the iterative models and `logs/`, which holds the run plan, an archive of the
source code, progress and resource budget. Finished runs are skipped. The iterative models train all 200 epochs
(no early stopping) and keep the checkpoint with the best validation NDCG@20.
`python scripts/run_experiments.py` runs the main grid: SGKR, SPHR and the three GNNs at four densities.

**Hardware.** SGKR and GF-CF run on a laptop CPU. SPHR and the graph baselines were trained on Kaggle GPUs
(`kaggle/kaggle_train.ipynb`). EASE inverts a dense 44,843 × 44,843 matrix and needs **about 9.1 GB of free
RAM**, so it was run in a Kaggle CPU session (`kaggle/kaggle_ease.ipynb`).

## Demo

```bash
python -m streamlit run app/streamlit_app.py
```

The interface is in Vietnamese and laid out for a 1920 × 1080 screen.

- **Gợi ý cho khách** (existing customer): the top-10 of all seven models side by side, with the customer's
  next purchase highlighted and SGKR's explanation of its first recommendation.
- **Khách mới** (new customer): build a history by searching for products. SGKR, SPHR, EASE and GF-CF answer at
  once; the graph baselines cannot serve unseen users.
- **Đánh giá mô hình** (evaluation): leaderboards, charts, statistical tests and report figures, split into
  the trained-model and the closed-form comparisons.

The demo loads the stored checkpoints and reads only the training and validation data.

## Models

| Model | Family | How it is obtained | Hardware |
|---|---|---|---|
| **SGKR** (proposed) | Item–item regression | Closed-form solve | CPU |
| **SPHR** (proposed) | History-based factorisation | Adam, full-catalog softmax | GPU |
| LightGCN [1] | Graph neural network | Gradient training, up to 200 epochs | GPU |
| XSimGCL [2] | Graph neural network + contrastive learning | Gradient training, up to 200 epochs | GPU |
| DirectAU [3] | Alignment–uniformity objective | Gradient training, up to 200 epochs | GPU |
| EASE [4] | Item–item regression | Closed-form solve | CPU |
| GF-CF [5] | Graph filtering | Closed-form (SVD) | CPU |

**SGKR** is an item–item ridge regression with graph regularisation. Each item is predicted from its 128 most
related items, with adaptive regularisation, and the system is solved exactly (Cholesky) with the target item
excluded from its own context. It has no user parameters and no optimiser, so it scores any purchase history
directly, including users never seen in training. It can also explain each recommendation by the history items
that contribute most.

**SPHR** represents a user by their normalised purchase history over diffusion-filtered item tables:

$$
\operatorname{score}(u, i) = \mathbf{q}_u^{\top} \tilde{\mathbf{v}}_i + b_i,
\qquad \mathbf{q}_u = |H_u|^{-1/2} \sum_{j \in H_u} \tilde{\mathbf{s}}_j
$$

where $H_u$ is the user's purchase history, $\tilde{\mathbf{s}}_j$ and $\tilde{\mathbf{v}}_i$ are the filtered source and target item
vectors and $b_i$ is an item bias. It is trained with Adam on a full-catalog softmax over leave-one-out contexts.
Like SGKR, it serves new users from their history alone.

The three graph baselines learn one embedding per training user, so they cannot recommend to a new user without
retraining.

## Data

Amazon Electronics 5-core reviews [6, 7]. Ratings of 4 or more are positives; users and items keep at least five
interactions; each user's history is split chronologically.

| Statistic | Value |
|---|---:|
| Users | 124,895 |
| Items | 44,843 |
| Training interactions | 858,192 |
| Validation interactions | 107,274 |
| Training sparsity | 99.98 % |
| History length, mean / median | 6.87 / 5 |

To measure robustness to data scarcity, every model is also trained on **75 %, 50 % and 25 %** of the training
interactions (sampled with seed 42).

## Results

### Accuracy

NDCG@20 on the validation split: ranking over the full catalog with each user's history masked, 107,215
evaluable users.

| Model | 25 % | 50 % | 75 % | 100 % | Training time at 100 % |
|---|---:|---:|---:|---:|---:|
| **SGKR** | 0.0080 | 0.0171 | 0.0293 | **0.0383** | 2.5 min (CPU) |
| **SPHR** | **0.0177** | **0.0227** | 0.0290 | 0.0352 | 127 min (GPU) |
| XSimGCL | 0.0112 | 0.0193 | 0.0276 | 0.0329 | 63 min (GPU) |
| DirectAU | 0.0103 | 0.0205 | 0.0270 | 0.0321 | 85 min (GPU) |
| LightGCN | 0.0084 | 0.0184 | 0.0226 | 0.0265 | 43 min (GPU) |
| EASE | 0.0067 | 0.0192 | **0.0294** | 0.0380 | 11 min (CPU) |
| GF-CF | 0.0039 | 0.0038 | 0.0084 | 0.0189 | 45 min (CPU) |

Training times were measured on different hardware (CPU vs. GPU) and are comparable only in order of magnitude.

<p align="center">
  <img src="reports/figures/H1_density_ndcg20.png" width="49%" alt="NDCG@20 by density, trained models">
  <img src="reports/figures/H1_closed_form_density_ndcg20.png" width="49%" alt="NDCG@20 by density, closed-form models">
</p>
<p align="center"><sub>NDCG@20 by training density. Left: SGKR, SPHR and the graph baselines. Right: SGKR and
the closed-form baselines.</sub></p>

### Statistical comparisons

The comparisons fall into two families, each with its own Holm correction. All tests are paired bootstraps over
users (10,000 resamples), one-sided in favour of the proposed model.

**Main comparison: proposed models vs. graph neural networks** (table B3, full density).

| Comparison | Δ NDCG@20 | 95 % CI | Holm p |
|---|---:|---|---:|
| SGKR − LightGCN | +0.0117 | [+0.0109, +0.0126] | 0.0006 |
| SGKR − XSimGCL | +0.0054 | [+0.0046, +0.0061] | 0.0006 |
| SGKR − DirectAU | +0.0062 | [+0.0055, +0.0070] | 0.0006 |
| SPHR − LightGCN | +0.0086 | [+0.0080, +0.0093] | 0.0006 |
| SPHR − XSimGCL | +0.0023 | [+0.0017, +0.0028] | 0.0006 |
| SPHR − DirectAU | +0.0031 | [+0.0025, +0.0038] | 0.0006 |

**Supplementary comparison: SGKR vs. closed-form models** (tables B5–B6, Holm over three comparisons per
density). This family was defined after the results were seen and is exploratory.

| SGKR − … | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|
| EASE (full, 7.5 GB) | SGKR better | EASE better | tie (p = 0.72) | tie (p = 0.051) |
| EASE top-128 (25–32 MB) | SGKR better | EASE better | SGKR better | SGKR better |
| GF-CF | SGKR better | SGKR better | SGKR better | SGKR better |

SGKR checkpoints take 6–41 MB depending on density, about the size of EASE top-128.

**Where SGKR does not win.** At 50 % density SGKR is below the three graph baselines and both EASE variants.
At 25 % and 50 % SPHR is the strongest model. GF-CF is weak on this very sparse data: its symmetric degree
normalisation favours rarely bought items.

## Evaluation protocol

- **Target:** each user's next purchase. Every model ranks the full catalog of 44,843 items, with the user's
  training history masked, and every model is evaluated on the same warm-user cohort.
- **Metrics:** NDCG@10/20, Recall@10/20 and MRR@20.
- **Model selection:** configurations were compared on one half of the validation users (hash split); the
  other half was reserved for confirmation. Decisions are logged append-only.
- **Significance:** paired bootstrap over users, one-sided, Holm-corrected within each comparison family.
- **Baseline configurations:** EASE and GF-CF use one fixed configuration each and are not tuned on this data:
  EASE λ = 300 (the middle of the usual 100/300/1000 grid) and GF-CF with the authors' defaults (k = 256,
  α = 0.3). The demo uses EASE pruned to its 128 largest weights per item; every EASE number above is the
  full matrix unless marked *top-128*.

## Limitations

- **One seed (42).** Variance across random seeds is not measured; the confidence intervals reflect user
  sampling only.
- **Validation split only.** The test split is held out, and no number here is a final test result.
- **Exploratory supplementary comparison.** The closed-form comparison was added after the results were known.
- **Untuned closed-form baselines.** EASE and GF-CF were not tuned on this data, so their scores may be lower
  bounds of what they can reach.
- **One dataset.** All conclusions concern Amazon Electronics.

## Repository layout

```
configs/   locked configurations: sgkr, sphr, lightgcn, xsimgcl, directau, ease, gfcf, common
src/       data pipeline, evaluator, losses, training loop, models (src/models/*.py)
scripts/   command-line entry points; scripts/verify/ holds SymPy checks of the closed forms
app/       Streamlit demo: streamlit_app.py (UI), recommender.py (model loading), benchmark.py (statistics)
kaggle/    notebooks used on Kaggle
results/   one folder per run and summary.csv (git tracks only checkpoints and training histories)
reports/   figures/ (PNG) and tables/ (CSV)
tests/     pytest suite
```

## References

1. X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, M. Wang. *LightGCN: Simplifying and Powering Graph Convolution
   Network for Recommendation.* SIGIR 2020.
2. J. Yu, X. Xia, T. Chen, L. Cui, N. Q. V. Hung, H. Yin. *XSimGCL: Towards Extremely Simple Graph Contrastive
   Learning for Recommendation.* IEEE TKDE 2023.
3. C. Wang, Y. Yu, W. Ma, M. Zhang, C. Chen, Y. Liu, S. Ma. *Towards Representation Alignment and Uniformity in
   Collaborative Filtering.* KDD 2022.
4. H. Steck. *Embarrassingly Shallow Autoencoders for Sparse Data.* WWW 2019.
5. Y. Shen, Y. Wu, Y. Zhang, C. Shan, J. Zhang, K. B. Letaief, D. Li. *How Powerful is Graph Convolution for
   Recommendation?* CIKM 2021.
6. R. He, J. McAuley. *Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class
   Collaborative Filtering.* WWW 2016.
7. J. McAuley, C. Targett, Q. Shi, A. van den Hengel. *Image-Based Recommendations on Styles and Substitutes.*
   SIGIR 2015.
