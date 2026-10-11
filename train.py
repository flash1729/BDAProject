"""Repeated k-fold evaluation of the deep multi-criteria CF model and baselines.

Examples:
    python train.py --smoke                       # 1 fold, 2 epochs, no baselines
    python train.py                               # 5 x 5-fold, all methods (Tables 5, 6)
    python train.py --resume                      # continue a killed run
    python train.py --eval-only --k 5             # recompute Table 6 from saved predictions
    python train.py --ablation emb --repeats 1    # Fig. 4 embedding-size sweep

Per (repeat, fold) results are written incrementally to results/runs/<run>/folds/*.json
and test predictions to results/runs/<run>/preds/*.npz.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _default_threads() -> int:
    for var in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        if os.environ.get(var, "").strip().isdigit() and int(os.environ[var]) > 0:
            return int(os.environ[var])
    return max(1, (os.cpu_count() or 4) - 2)


THREADS = _default_threads()
# Must be set before numpy / TensorFlow are imported.
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
# 3 hides a harmless C++ "use_unbounded_threadpool" message logged on every fit; Python errors still raise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import metrics as M  # noqa: E402
from data import CRITERIA, CRITERIA_LABELS  # noqa: E402
from paper_values import (METHOD_LABELS, PAPER_TABLE3, PAPER_TABLE4, PAPER_TABLE5,  # noqa: E402
                          PAPER_TABLE6)

log = logging.getLogger("train")

BATCH_SIZE = 512
BASELINES = ["single_dnn", "svd", "svdpp", "slopeone", "baselineonly", "knn"]
SKIP_ALIASES = {"svd++": "svdpp", "knnbaseline": "knn", "baseline": "baselineonly", "single": "single_dnn",
                "singledn": "single_dnn", "singlednn": "single_dnn"}
TABLE6_METRICS = ["mae", "f1", "f2", "fcp", "map", "mrr", "precision", "recall"]
METRIC_LABELS = {"mae": "MAE", "f1": "F1", "f2": "F2", "fcp": "FCP", "map": "MAP", "mrr": "MRR",
                 "precision": "Precision", "recall": "Recall"}

ABLATIONS = {
    # name -> {method_name: kwargs for run_ours}
    "emb": {f"emb{d}": {"emb_dim": d} for d in [8, 16, 32, 64, 128]},
    "layers": {
        "-".join(map(str, h)): {"hidden": h}
        for h in [(8,), (16, 8), (32, 16, 8), (64, 32, 16, 8), (128, 64, 32, 16, 8), (256, 128, 64, 32, 16, 8)]
    },
    "optimizer": {o: {"optimizer": o} for o in ["adam", "adagrad", "rmsprop", "adamax", "adadelta", "sgd"]},
}

# Only these settings change what gets trained; resuming with different values is refused.
TRAINING_KEYS = ["folds", "epochs", "patience", "seed", "val_frac", "ablation", "data", "max_restarts"]


# --------------------------------------------------------------------------- runtime

def configure_runtime(threads: int) -> str:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(min(2, threads))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:  # only present when tensorflow-metal is installed on Apple Silicon
        device = f"Metal GPU ({len(gpus)} device(s), tensorflow-metal)"
    else:
        tf.config.experimental.enable_op_determinism()
        device = "CPU"
    log.info("TensorFlow %s | device: %s | intra-op threads: %d | machine: %s %s",
             tf.__version__, device, threads, platform.system(), platform.machine())
    return device


def seed_everything(seed: int) -> None:
    import keras

    keras.utils.set_random_seed(seed)  # python random, numpy, tensorflow


def fmt_secs(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- folds

@dataclass
class Fold:
    repeat: int
    fold: int
    train: pd.DataFrame  # 80% minus validation
    val: pd.DataFrame  # 1% of the 80%
    test: pd.DataFrame  # 20%
    n_users: int
    n_items: int
    seed: int

    @property
    def tag(self) -> str:
        return f"r{self.repeat}_f{self.fold}"


def iter_folds(df: pd.DataFrame, args):
    n_users, n_items = int(df["user_idx"].max()) + 1, int(df["item_idx"].max()) + 1
    for r in range(args.repeats):
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + r)
        for f, (train_idx, test_idx) in enumerate(kf.split(df)):
            if args.smoke and f > 0:
                break
            rng = np.random.default_rng(args.seed + 1000 * r + f)
            perm = rng.permutation(train_idx)
            n_val = max(1, int(round(args.val_frac * len(perm))))
            yield Fold(r, f, df.iloc[np.sort(perm[n_val:])], df.iloc[np.sort(perm[:n_val])], df.iloc[test_idx],
                       n_users, n_items, args.seed + 1000 * r + 10 * f)


def ui(d: pd.DataFrame):
    return [d["user_idx"].to_numpy(np.int32), d["item_idx"].to_numpy(np.int32)]


# --------------------------------------------------------------------------- methods

def fit_keras(model, x, y, x_val, y_val, args) -> dict:
    import keras

    t0 = time.time()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    h = model.fit(x, y, validation_data=(x_val, y_val), epochs=args.epochs, batch_size=BATCH_SIZE,
                  shuffle=True, verbose=0, callbacks=[es])
    vl = h.history["val_loss"]
    best = int(np.argmin(vl))
    return {"epochs_run": len(vl), "best_epoch": best + 1, "best_val_mae": float(vl[best]),
            "final_train_mae": float(h.history["loss"][-1]), "fit_seconds": round(time.time() - t0, 2)}


def predict(model, x) -> np.ndarray:
    return model.predict(x, batch_size=8192, verbose=0)


def fit_with_restarts(build, x, y, x_val, y_val, args, seed: int, label: str):
    """Build + fit a model; if any ReLU output unit is dead (0 for every training row), re-initialise
    with a new seed and retrain, up to --max-restarts times. The paper's ReLU output layer with
    N(0, 0.05) init occasionally kills an output unit, which can never recover (zero gradient)."""
    dead_history = []
    for attempt in range(args.max_restarts + 1):
        seed_everything(seed + 7919 * attempt)
        model = build()
        info = fit_keras(model, x, y, x_val, y_val, args)
        dead = np.flatnonzero(predict(model, x).reshape(len(y), -1).max(axis=0) <= 0).tolist()
        if not dead:
            break
        dead_history.append(dead)
        log.warning("    %s: dead ReLU output unit(s) %s after attempt %d%s", label, dead, attempt + 1,
                    ", re-initialising" if attempt < args.max_restarts else ", giving up (keeping it)")
    info.update({"restarts": attempt,
                 "dead_units_per_attempt": dead_history, "dead_units_final": dead})
    return model, info


def run_ours(fold: Fold, args, emb_dim=64, hidden=None, optimizer="adam") -> tuple[dict, np.ndarray]:
    """Criteria DNN + overall DNN, trained independently, chained at inference."""
    from model import CRITERIA_HIDDEN, build_criteria_dnn, build_overall_dnn

    hidden = tuple(hidden or CRITERIA_HIDDEN)
    y_crit = {k: fold.__dict__[k][CRITERIA].to_numpy(np.float32) for k in ("train", "val", "test")}

    # net 1: (user, item) -> 7 true criteria
    net1, info1 = fit_with_restarts(
        lambda: build_criteria_dnn(fold.n_users, fold.n_items, emb_dim=emb_dim, hidden=hidden, optimizer=optimizer),
        ui(fold.train), y_crit["train"], ui(fold.val), y_crit["val"], args, fold.seed + 1, "criteria DNN")
    pred_crit = predict(net1, ui(fold.test))
    crit_mae = {c: M.mae(y_crit["test"][:, j], pred_crit[:, j]) for j, c in enumerate(CRITERIA)}
    crit_mae["mean"] = float(np.mean([crit_mae[c] for c in CRITERIA]))

    # net 2: z-scored true criteria -> true overall (scaler fit on the training part only)
    scaler = StandardScaler().fit(y_crit["train"])
    y_over = {k: fold.__dict__[k]["overall"].to_numpy(np.float32) for k in ("train", "val", "test")}
    net2, info2 = fit_with_restarts(build_overall_dnn, scaler.transform(y_crit["train"]), y_over["train"],
                                    scaler.transform(y_crit["val"]), y_over["val"], args, fold.seed + 2, "overall DNN")

    # chain: predicted criteria -> scaler -> net 2
    pred_overall = predict(net2, scaler.transform(pred_crit)).ravel()
    oracle = predict(net2, scaler.transform(y_crit["test"])).ravel()  # diagnostic: net 2 on true criteria
    extra = {"criteria_mae": crit_mae, "net1": info1, "net2": info2,
             "net2_on_true_criteria_mae": M.mae(y_over["test"], oracle)}
    return extra, pred_overall


def run_single_dnn(fold: Fold, args) -> tuple[dict, np.ndarray]:
    from model import build_single_dnn

    y = {k: fold.__dict__[k]["overall"].to_numpy(np.float32) for k in ("train", "val")}
    net, info = fit_with_restarts(lambda: build_single_dnn(fold.n_users, fold.n_items), ui(fold.train), y["train"],
                                  ui(fold.val), y["val"], args, fold.seed + 3, "single DNN")
    return {"net": info}, predict(net, ui(fold.test)).ravel()


def run_surprise(name: str, fold: Fold, args) -> tuple[dict, np.ndarray]:
    from surprise import SVD, BaselineOnly, Dataset, KNNBaseline, Reader, SlopeOne, SVDpp

    seed = fold.seed + 10 + BASELINES.index(name)
    np.random.seed(seed)
    algo = {
        "svd": lambda: SVD(random_state=seed),
        "svdpp": lambda: SVDpp(random_state=seed),
        "slopeone": lambda: SlopeOne(),
        "baselineonly": lambda: BaselineOnly(verbose=False),
        # Item-based: user-based would need a 72k x 72k similarity matrix (~40 GB).
        "knn": lambda: KNNBaseline(sim_options={"name": "pearson_baseline", "user_based": False}, verbose=False),
    }[name]()
    # Baselines need no validation set, so they train on the full 80% (train + val).
    train = pd.concat([fold.train, fold.val])
    t0 = time.time()
    trainset = Dataset.load_from_df(train[["user_idx", "item_idx", "overall"]], Reader(rating_scale=(1, 5)))
    algo.fit(trainset.build_full_trainset())
    fit_s = time.time() - t0
    users, items = fold.test["user_idx"].tolist(), fold.test["item_idx"].tolist()
    pred = np.array([algo.predict(u, i).est for u, i in zip(users, items)], dtype=np.float64)
    return {"fit_seconds": round(fit_s, 2)}, pred


def build_methods(args) -> dict:
    """name -> callable(fold, args) -> (extra_info, predicted_overall)."""
    if args.ablation:
        return {name: (lambda kw: lambda fold, a: run_ours(fold, a, **kw))(kw)
                for name, kw in ABLATIONS[args.ablation].items()}
    methods = {"ours": run_ours}
    for b in BASELINES:
        if b in args.skip:
            continue
        methods[b] = run_single_dnn if b == "single_dnn" else (lambda n: lambda fold, a: run_surprise(n, fold, a))(b)
    return methods


# --------------------------------------------------------------------------- run / resume

def run_paths(args) -> tuple[Path, Path, Path]:
    name = f"ablation_{args.ablation}" if args.ablation else "main"
    name = f"smoke_{name}" if args.smoke else name
    run_dir = Path(args.results) / "runs" / name
    return run_dir, run_dir / "folds", run_dir / "preds"


def prepare_run_dir(args, run_dir: Path, fold_dir: Path, pred_dir: Path) -> None:
    cfg_path = run_dir / "config.json"
    cfg = {k: getattr(args, k) for k in TRAINING_KEYS}
    existing = list(fold_dir.glob("*.json")) if fold_dir.exists() else []
    if existing and not (args.resume or args.fresh or args.smoke):
        raise SystemExit(f"{run_dir} already has {len(existing)} fold result(s). "
                         "Use --resume to continue it or --fresh to delete it and start over.")
    if (args.fresh or args.smoke) and run_dir.exists():
        shutil.rmtree(run_dir)
    if args.resume and cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {k: (old[k], cfg[k]) for k in TRAINING_KEYS if k in old and old[k] != cfg[k]}
        if diff:
            raise SystemExit(f"--resume with different training settings (old, new): {diff}. "
                             "Use the original flags or --fresh.")
    fold_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for m in args.rerun:
        n = 0
        for fp in fold_dir.glob("r*_f*.json"):
            rec = json.loads(fp.read_text())
            if rec["methods"].pop(m, None) is not None:
                write_json_atomic(fp, rec)
                n += 1
            (pred_dir / f"{fp.stem}_{m}.npz").unlink(missing_ok=True)
        log.info("--rerun: discarded saved '%s' results from %d fold(s)", m, n)
    write_json_atomic(cfg_path, cfg)


def train_all(df: pd.DataFrame, args, fold_dir: Path, pred_dir: Path) -> None:
    methods = build_methods(args)
    folds = list(iter_folds(df, args))
    n_units = len(folds)

    def done(fold, name, rec):
        return name in rec.get("methods", {}) and (pred_dir / f"{fold.tag}_{name}.npz").exists()

    def load(fold):
        p = fold_dir / f"{fold.tag}.json"
        return json.loads(p.read_text()) if p.exists() else {"repeat": fold.repeat, "fold": fold.fold, "methods": {}}

    pending = [f for f in folds if not all(done(f, m, load(f)) for m in methods)]
    log.info("methods: %s", ", ".join(methods))
    log.info("%d repeat(s) x %d fold(s) = %d fold run(s); %d already complete, %d to run",
             args.repeats, 1 if args.smoke else args.folds, n_units, n_units - len(pending), len(pending))

    t_start = time.time()
    for k, fold in enumerate(pending, 1):
        t_fold = time.time()
        rec = load(fold)
        log.info("[%d/%d] %s  train=%d val=%d test=%d", k, len(pending), fold.tag,
                 len(fold.train), len(fold.val), len(fold.test))
        for name, fn in methods.items():
            if done(fold, name, rec):
                log.info("    %-14s already done, skipping", name)
                continue
            t0 = time.time()
            extra, pred = fn(fold, args)
            y = fold.test["overall"].to_numpy()
            u = fold.test["user_idx"].to_numpy()
            np.savez_compressed(pred_dir / f"{fold.tag}_{name}.npz", user=u,
                                item=fold.test["item_idx"].to_numpy(), true=y, pred=pred)
            res = {**extra, "metrics": M.overall_metrics(u, y, pred, args.k, args.rel_threshold, args.topk_mode),
                   "seconds": round(time.time() - t0, 2)}
            rec["methods"][name] = res
            write_json_atomic(fold_dir / f"{fold.tag}.json", rec)
            m = res["metrics"]
            msg = (f"    {name:<14s} MAE {m['mae']:.4f}  F1 {m['f1']:.4f}  F2 {m['f2']:.4f}  FCP {m['fcp']:.4f}"
                   f"  MAP {m['map']:.4f}  MRR {m['mrr']:.4f}  ({fmt_secs(res['seconds'])})")
            if "criteria_mae" in res:
                msg += (f"\n    {'':<14s} criteria MAE mean {res['criteria_mae']['mean']:.4f}"
                        f" | net1 best epoch {res['net1']['best_epoch']}/{res['net1']['epochs_run']}"
                        f" | net2 best epoch {res['net2']['best_epoch']}/{res['net2']['epochs_run']}"
                        f" | net2 on true criteria MAE {res['net2_on_true_criteria_mae']:.4f}"
                        f" | restarts net1 {res['net1']['restarts']} net2 {res['net2']['restarts']}")
            log.info(msg)
        elapsed = time.time() - t_start
        eta = elapsed / k * (len(pending) - k)
        log.info("[%d/%d] %s done in %s | elapsed %s | ETA %s", k, len(pending), fold.tag,
                 fmt_secs(time.time() - t_fold), fmt_secs(elapsed), fmt_secs(eta))


# --------------------------------------------------------------------------- aggregation

def collect(args, fold_dir: Path, pred_dir: Path, method_names) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute overall metrics from saved predictions with the current --k/--rel-threshold."""
    rows, crit_rows, diag_rows = [], [], []
    for p in sorted(fold_dir.glob("r*_f*.json")):
        rec = json.loads(p.read_text())
        tag = p.stem
        for name, res in rec["methods"].items():
            if name not in method_names:
                continue
            npz = pred_dir / f"{tag}_{name}.npz"
            if not npz.exists():
                continue
            d = np.load(npz)
            m = M.overall_metrics(d["user"], d["true"], d["pred"], args.k, args.rel_threshold, args.topk_mode)
            rows.append({"run": tag, "method": name, **{k: m[k] for k in TABLE6_METRICS}})
            if "criteria_mae" in res:
                crit_rows.append({"run": tag, "method": name, **res["criteria_mae"]})
                diag_rows.append({"run": tag, "method": name,
                                  "net2_on_true_criteria_mae": res["net2_on_true_criteria_mae"],
                                  "net1_best_epoch": res["net1"]["best_epoch"],
                                  "net2_best_epoch": res["net2"]["best_epoch"],
                                  "net1_restarts": res["net1"].get("restarts", 0),
                                  "net2_restarts": res["net2"].get("restarts", 0)})
    return pd.DataFrame(rows), pd.DataFrame(crit_rows), pd.DataFrame(diag_rows)


def mean_std(s: pd.Series) -> tuple[float, float]:
    return float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0


def pm(mean, std, nd=4) -> str:
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def write_main_tables(args, runs: pd.DataFrame, crit: pd.DataFrame, diag: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in ["ours", *BASELINES] if m in set(runs["method"])]
    n_runs = runs.groupby("method").size().to_dict()

    # Table 5
    t5 = []
    ours_crit = crit[crit["method"] == "ours"]
    for c in [*CRITERIA, "mean"]:
        mu, sd = mean_std(ours_crit[c])
        t5.append({"criterion": CRITERIA_LABELS.get(c, "Mean"), "mae_mean": mu, "mae_std": sd,
                   "n_runs": len(ours_crit), "paper_mean": PAPER_TABLE5[c][0], "paper_std": PAPER_TABLE5[c][1]})
    t5 = pd.DataFrame(t5)
    t5.to_csv(out_dir / "table5.csv", index=False, float_format="%.4f")

    # Table 6 (long format)
    t6 = []
    for meth in methods:
        sub = runs[runs["method"] == meth]
        for met in TABLE6_METRICS:
            mu, sd = mean_std(sub[met])
            paper = PAPER_TABLE6.get(meth, {}).get(met, (np.nan, np.nan))
            t6.append({"method": meth, "metric": met, "mean": mu, "std": sd, "n_runs": len(sub),
                       "paper_mean": paper[0], "paper_std": paper[1]})
    t6 = pd.DataFrame(t6)
    t6.to_csv(out_dir / "table6.csv", index=False, float_format="%.4f")

    # Markdown comparison
    get = lambda meth, met: t6[(t6.method == meth) & (t6.metric == met)].iloc[0]  # noqa: E731
    L = ["# Table 6: paper vs ours", "",
         f"Settings: {args.repeats} repeat(s) x {args.folds}-fold CV"
         f"{' (smoke: first fold only)' if args.smoke else ''}, max {args.epochs} epochs, early stopping "
         f"patience {args.patience} on a {args.val_frac:.0%} validation split, K={args.k}, "
         f"relevant = true overall >= {args.rel_threshold:g}, top-K mode `{args.topk_mode}`. "
         "Mean ± sample std over all (repeat, fold) runs.", "",
         "## Our model (chained criteria DNN -> overall DNN)", "",
         "| Metric | Paper | Ours | Ours - Paper |", "|---|---:|---:|---:|"]
    if "ours" in methods:
        for met in ["mae", "f1", "f2", "fcp", "map", "mrr"]:
            r = get("ours", met)
            L.append(f"| {METRIC_LABELS[met]} | {pm(r.paper_mean, r.paper_std)} | {pm(r['mean'], r['std'])} "
                     f"| {r['mean'] - r.paper_mean:+.4f} |")
    L += ["", "## All methods, ours (paper in brackets)", "",
          "| Metric | " + " | ".join(METHOD_LABELS[m] for m in methods) + " |",
          "|---|" + "---:|" * len(methods)]
    for met in TABLE6_METRICS:
        cells = []
        for meth in methods:
            r = get(meth, met)
            paper = f" ({r.paper_mean:.4f})" if not np.isnan(r.paper_mean) else ""
            cells.append(f"{pm(r['mean'], r['std'])}{paper}")
        L.append(f"| {METRIC_LABELS[met]} | " + " | ".join(cells) + " |")
    L.append("| runs | " + " | ".join(str(n_runs[m]) for m in methods) + " |")

    # ranking of MAE: does ours beat the baselines, as the paper claims?
    if "ours" in methods and len(methods) > 1:
        L += ["", "## Who wins (ours vs best baseline)", "", "| Metric | Best baseline | Ours | Ours better? |",
              "|---|---|---:|:---:|"]
        for met in ["mae", "f1", "f2", "fcp", "map", "mrr"]:
            sub = t6[(t6.metric == met) & (t6.method != "ours")]
            best = sub.loc[sub["mean"].idxmin() if met == "mae" else sub["mean"].idxmax()]
            ours = get("ours", met)["mean"]
            better = ours < best["mean"] if met == "mae" else ours > best["mean"]
            L.append(f"| {METRIC_LABELS[met]} | {METHOD_LABELS[best.method]} {best['mean']:.4f} | {ours:.4f} "
                     f"| {'yes' if better else 'no'} |")

    L += ["", "## Table 5: criteria DNN per-criterion MAE", "", "| Criterion | Paper | Ours | Ours - Paper |",
          "|---|---:|---:|---:|"]
    for r in t5.itertuples():
        L.append(f"| {r.criterion} | {pm(r.paper_mean, r.paper_std)} | {pm(r.mae_mean, r.mae_std)} "
                 f"| {r.mae_mean - r.paper_mean:+.4f} |")

    if len(diag):
        d = diag[diag.method == "ours"]
        L += ["", "## Diagnostics (ours)", "",
              f"- Overall DNN applied to the *true* test criteria: MAE {pm(*mean_std(d.net2_on_true_criteria_mae))} "
              "(lower bound for the chained model if net 1 were perfect).",
              f"- Best epoch chosen by early stopping: criteria DNN {d.net1_best_epoch.mean():.1f} "
              f"(range {d.net1_best_epoch.min()}-{d.net1_best_epoch.max()}), overall DNN "
              f"{d.net2_best_epoch.mean():.1f} (range {d.net2_best_epoch.min()}-{d.net2_best_epoch.max()}).",
              f"- Dead-ReLU-output restarts (`--max-restarts {args.max_restarts}`): criteria DNN restarted in "
              f"{int((d.net1_restarts > 0).sum())}/{len(d)} runs ({int(d.net1_restarts.sum())} restarts total), "
              f"overall DNN in {int((d.net2_restarts > 0).sum())}/{len(d)} runs. The paper's ReLU output layer "
              "can kill an output unit for good; such a run is re-initialised with a new seed."]

    L += ["", "## Notes on definitions", "",
          "- The paper does not state K, the relevance threshold, or what counts as 'recommended'. "
          "See `metrics.py` for the definitions used here; `--topk-mode plain` is the alternative.",
          "- MAP@K and MRR@K: most test users have a single test item, so with eq. 19 as written AP equals RR "
          "for them and MAP is close to MRR. The paper's much lower MAP implies an unstated variant.",
          "- FCP counts prediction ties as discordant (eq. 18), so methods that predict a constant for unseen "
          "users (e.g. SlopeOne) get a very low FCP.",
          "- KNN with baseline is item-based (`pearson_baseline`); user-based would need a ~40 GB similarity matrix.",
          "- Surprise baselines train on the full 80% training split; the DNNs hold out 1% of it for early stopping.",
          "- Data: dedupe on (user, hotel) removes 33 repeat stays, so we have 81,052 ratings vs the paper's 81,085.",
          ""]
    (out_dir / "table6_paper_vs_ours.md").write_text("\n".join(L))
    log.info("wrote %s, %s, %s", out_dir / "table5.csv", out_dir / "table6.csv", out_dir / "table6_paper_vs_ours.md")
    print("\n" + "\n".join(L))


def write_ablation_table(args, runs: pd.DataFrame, out_dir: Path) -> None:
    paper = {"optimizer": PAPER_TABLE3, "layers": PAPER_TABLE4}.get(args.ablation, {})
    rows = []
    for name in ABLATIONS[args.ablation]:
        sub = runs[runs.method == name]
        if not len(sub):
            continue
        mu, sd = mean_std(sub["mae"])
        p = paper.get(name, (np.nan, np.nan))
        rows.append({"config": name, "mae_mean": mu, "mae_std": sd, "n_runs": len(sub),
                     "paper_mean": p[0], "paper_std": p[1],
                     **{f"{m}_mean": sub[m].mean() for m in ["f1", "f2", "fcp", "map", "mrr"]}})
    t = pd.DataFrame(rows)
    path = out_dir / f"ablation_{args.ablation}.csv"
    t.to_csv(path, index=False, float_format="%.4f")
    log.info("wrote %s\n%s", path, t[["config", "mae_mean", "mae_std", "n_runs", "paper_mean"]].to_string(index=False))


# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ratings.parquet")
    ap.add_argument("--results", default="results")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50, help="max epochs (early stopping picks the best)")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience on validation MAE")
    ap.add_argument("--val-frac", type=float, default=0.01, help="fraction of the training fold held out")
    ap.add_argument("--k", type=int, default=10, help="top-K cutoff for F1/F2/MAP/MRR")
    ap.add_argument("--rel-threshold", type=float, default=4.0, help="relevant if true overall >= this")
    ap.add_argument("--topk-mode", choices=["surprise", "plain"], default="surprise",
                    help="'surprise': recommended = top-K with predicted >= threshold; 'plain': top-K")
    ap.add_argument("--skip-baselines", default="", help=f"comma list from {','.join(BASELINES)} or 'all'")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, no baselines")
    ap.add_argument("--resume", action="store_true", help="continue a previous run, skipping finished work")
    ap.add_argument("--fresh", action="store_true", help="delete previous results of this run and start over")
    ap.add_argument("--eval-only", action="store_true", help="only rebuild tables from saved predictions")
    ap.add_argument("--ablation", choices=sorted(ABLATIONS), help="run a sweep instead of the main comparison")
    ap.add_argument("--max-restarts", type=int, default=5,
                    help="re-initialise a DNN whose ReLU output unit died, up to this many times (0 = never)")
    ap.add_argument("--rerun", default="", help="comma list of methods whose saved results are discarded and "
                    "recomputed; implies --resume (e.g. --rerun ours)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=THREADS,
                    help="TF intra-op threads (default: $OMP_NUM_THREADS or cpu_count-2)")
    args = ap.parse_args(argv)

    if args.smoke:
        args.repeats, args.epochs = 1, 2
        args.skip_baselines = "all"
    skip = {SKIP_ALIASES.get(s.strip().lower(), s.strip().lower()) for s in args.skip_baselines.split(",") if s.strip()}
    if "all" in skip:
        skip = set(BASELINES)
    unknown = skip - set(BASELINES)
    if unknown:
        ap.error(f"unknown baseline(s) in --skip-baselines: {sorted(unknown)}; choose from {BASELINES}")
    args.skip = sorted(skip)
    args.rerun = [m.strip() for m in args.rerun.split(",") if m.strip()]
    if args.rerun:
        args.resume = True
    if args.resume and args.fresh:
        ap.error("--resume and --fresh are mutually exclusive")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    run_dir, fold_dir, pred_dir = run_paths(args)
    out_dir = Path(args.results) / "smoke" if args.smoke else Path(args.results)
    log.info("command: %s", " ".join(sys.argv))

    if not args.eval_only:
        configure_runtime(args.threads)
        df = pd.read_parquet(args.data)
        log.info("data: %d ratings, %d users, %d items", len(df), df.user_idx.nunique(), df.item_idx.nunique())
        prepare_run_dir(args, run_dir, fold_dir, pred_dir)
        t0 = time.time()
        train_all(df, args, fold_dir, pred_dir)
        log.info("training finished in %s", fmt_secs(time.time() - t0))
    elif not fold_dir.exists():
        raise SystemExit(f"nothing to evaluate in {fold_dir}")

    runs, crit, diag = collect(args, fold_dir, pred_dir, set(build_methods(args)) | {"ours", *BASELINES})
    if not len(runs):
        raise SystemExit("no finished folds found")
    if args.ablation:
        write_ablation_table(args, runs, out_dir)
    else:
        write_main_tables(args, runs, crit, diag, out_dir)


if __name__ == "__main__":
    main()
