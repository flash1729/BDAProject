# Deep multi-criteria collaborative filtering (reproduction)

Reproduction of Nassar, Jafar, Rahhal, *A novel deep multi-criteria collaborative filtering
model for recommendation system*, Knowledge-Based Systems 187 (2020) 104811,
doi:10.1016/j.knosys.2019.06.019, on Hongning Wang's LARA TripAdvisor dataset
(`hotel_<ID>_parsed.txt` files, anywhere under this folder).

**Full write-up:** open [`report.html`](report.html) in a browser. It covers the problem, the model, the method, what was built, and the results compared with the paper.

## Data

The dataset is not in this repo. Download the LARA TripAdvisor hotel reviews (Hongning Wang,
`hotel_<ID>_parsed.txt` files) and put them anywhere under this folder, e.g. `Texts/`.
`data.py` finds them recursively. `data/` and the per-fold prediction files are git-ignored and
can be regenerated with `python data.py` and `./run.sh`.

## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

(or `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`). Runs on CPU.
If `tensorflow-metal` is installed on Apple Silicon it is used automatically, but it isn't required.

## Run

```bash
.venv/bin/python data.py              # parse + filter -> data/ratings.parquet, prints Tables 1 and 2
.venv/bin/python -m pytest -q tests   # parser + metric unit tests
.venv/bin/python train.py --smoke     # 1 fold, 2 epochs, no baselines (~20 s)
./run.sh                              # full 5 x 5-fold run with all baselines
./run.sh --resume                     # continue a killed run; finished folds/methods are skipped
```

`run.sh` wraps `train.py` in `caffeinate -i nice -n 10` (the Mac stays awake, interactive apps
keep priority) and appends output to `results/train.log`. TensorFlow uses `$OMP_NUM_THREADS`
threads if set, else `cpu_count - 2`.

Useful flags (`python train.py -h` for all):

| flag | default | meaning |
|---|---|---|
| `--repeats`, `--folds` | 5, 5 | repeated k-fold CV |
| `--epochs`, `--patience` | 50, 5 | max epochs; early stopping on the 1% validation split restores the best epoch |
| `--k`, `--rel-threshold` | 10, 4 | top-K cutoff; relevant = true overall >= threshold |
| `--topk-mode` | `surprise` | what counts as "recommended" for precision/recall (see `metrics.py`) |
| `--skip-baselines` | none | e.g. `svdpp,knn`, or `all` (names: single_dnn, svd, svdpp, slopeone, baselineonly, knn) |
| `--resume` / `--fresh` | | continue / wipe an existing run |
| `--rerun ours` | | discard one method's saved results and recompute only it (implies `--resume`) |
| `--max-restarts` | 5 | re-initialise a DNN whose ReLU output unit died (0 = never, strictly as in the paper) |
| `--eval-only` | | rebuild tables from saved predictions, e.g. with a different `--k` |
| `--ablation emb\|layers\|optimizer` | | Fig. 4 / Table 4 / Table 3 sweeps (criteria DNN only) |

## Outputs

- `results/data_stats.md`: Tables 1 and 2, paper vs ours
- `results/table5.csv`: per-criterion MAE of the criteria DNN
- `results/table6.csv`: overall-rating metrics for every method (mean, std, paper values)
- `results/table6_paper_vs_ours.md`: side-by-side comparison with the paper
- `results/runs/<run>/folds/r<repeat>_f<fold>.json`: per-fold results, written as each method finishes
- `results/runs/<run>/preds/*.npz`: test predictions per fold and method (tables are always
  recomputed from these, so `--k` / `--rel-threshold` can be changed with `--eval-only`)
- `results/ablation_<name>.csv`: sweep results

## Files

- `data.py`: regex parser and filtering (exactly 8 ratings -> all in 1..5 -> drop anonymous -> dedupe (user, hotel))
- `model.py`: criteria DNN, overall DNN, single-DNN baseline (Keras 3)
- `metrics.py`: MAE, F1/F2@K, FCP, MAP@K, MRR@K
- `train.py`: CV loop, baselines (`surprise`), resume, tables, ablations
- `paper_values.py`: the paper's reported numbers

## Choices the paper leaves open

- **Data.** Filtering to complete 1..5 ratings gives exactly the paper's 81,085 ratings,
  72,119 users and 1,850 hotels. The (user, hotel) dedupe then removes 33 repeat stays, so we
  train on 81,052. Anonymous "A TripAdvisor Member" reviews never have all 8 ratings, so step 3 removes nothing.
- **Precision/recall@K.** "Recommended" = the top-K items by prediction *whose predicted rating is
  >= the threshold*, with precision/recall = 1 when undefined (the 2019 `surprise` FAQ recipe).
  With this definition the surprise baselines reproduce the paper's MAE/F1/F2 to about 3 decimals.
- **MAP/FCP.** Implemented as written in eq. 14-19. They don't match the paper's
  absolute values (see the notes in `results/table6_paper_vs_ours.md`).
- **Epochs.** Not given in the paper. We train up to `--epochs` and keep the best epoch on the 1% validation split.
- **Baselines.** `surprise` defaults. KNNBaseline is item-based with `pearson_baseline`, because
  user-based needs a 72k x 72k similarity matrix. The baselines train on the full 80% split.
- **Dead ReLU outputs.** The paper's ReLU output layer with N(0, 0.05) init sometimes kills an
  output unit: it outputs 0 for every input and, with zero gradient, never recovers. In our first
  run 1 of 5 folds lost the "Value" unit, which turned the chained MAE from about 0.76 into 2.34.
  After training, any DNN with an output unit that is 0 on every training row is re-initialised
  with a new seed (`--max-restarts`, default 5). Restarts are logged and counted in the results.
- **mean ± std** is the sample std over all repeat x fold runs (25 by default).
- **Optimizer sweep.** Uses Keras 2.2's default learning rate for each optimizer, and Adam at 0.001 as in the paper.
