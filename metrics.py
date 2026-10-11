"""Overall-rating metrics from Section 4.2 (eq. 10-20).

All top-K metrics rank each user's *test* items by predicted rating and look at the
first K. An item is relevant when its true rating >= rel_threshold.

The paper does not define "recommended" precisely. Its baselines are the `surprise`
library's algorithms, and the published F-scores (e.g. SlopeOne F2 0.92 vs SVD 0.80)
are only explainable if "recommended" also requires the *predicted* rating to clear
the threshold, as in the surprise FAQ `precision_recall_at_k` recipe of that era.
So by default (topk_mode="surprise"):
    recommended_u = top-K items of u with predicted >= rel_threshold
    precision_u   = |rel & rec| / |rec|   (1 if nothing recommended)
    recall_u      = |rel & rec| / |rel|   (1 if u has no relevant item)
topk_mode="plain" uses recommended_u = top-K items regardless of prediction.
F_beta is computed from the user-averaged precision and recall (eq. 11-13).

MAP@K (eq. 19) and MRR@K (eq. 20) use the plain top-K ranking; users with no relevant
test item contribute 0. FCP follows eq. 14-18 over ordered item pairs per user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def fcp(users, y_true, y_pred) -> float:
    """Fraction of Concordant Pairs, nc and nd summed over users (eq. 14-18)."""
    df = pd.DataFrame({"u": users, "t": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    sizes = df.groupby("u")["t"].transform("size")
    df = df[sizes >= 2]
    nc = nd = 0
    for _, g in df.groupby("u", sort=False):
        t, p = g["t"].to_numpy(), g["p"].to_numpy()
        t_gt = t[:, None] > t[None, :]
        nc += int(np.sum((p[:, None] > p[None, :]) & t_gt))
        nd += int(np.sum((p[:, None] >= p[None, :]) & t_gt.T))  # r_ui < r_uj
    return nc / (nc + nd) if nc + nd else float("nan")


def ranking_metrics(users, y_true, y_pred, k: int = 10, rel_threshold: float = 4.0,
                    topk_mode: str = "surprise") -> dict[str, float]:
    df = pd.DataFrame({"u": np.asarray(users), "t": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    # rank by predicted rating desc within user; stable sort keeps ties in input order
    df = df.iloc[np.lexsort((-df["p"].to_numpy(), df["u"].to_numpy()))].reset_index(drop=True)
    df["rank"] = df.groupby("u").cumcount() + 1
    df["rel"] = df["t"] >= rel_threshold
    df["topk"] = df["rank"] <= k
    if topk_mode == "surprise":
        df["rec"] = df["topk"] & (df["p"] >= rel_threshold)
    elif topk_mode == "plain":
        df["rec"] = df["topk"]
    else:
        raise ValueError(topk_mode)
    df["rel_rec"] = df["rel"] & df["rec"]
    df["rel_topk"] = df["rel"] & df["topk"]
    df["cum_rel"] = df.groupby("u")["rel_topk"].cumsum()
    df["prec_at_n"] = np.where(df["rel_topk"], df["cum_rel"] / df["rank"], 0.0)
    df["rr"] = np.where(df["rel_topk"], 1.0 / df["rank"], 0.0)

    g = df.groupby("u").agg(
        n_rel=("rel", "sum"), n_rec=("rec", "sum"), n_rel_rec=("rel_rec", "sum"),
        ap_sum=("prec_at_n", "sum"), rr=("rr", "max"),
    )
    precision_u = np.where(g["n_rec"] > 0, g["n_rel_rec"] / g["n_rec"].clip(lower=1), 1.0)
    recall_u = np.where(g["n_rel"] > 0, g["n_rel_rec"] / g["n_rel"].clip(lower=1), 1.0)
    precision, recall = float(precision_u.mean()), float(recall_u.mean())

    def f_beta(beta):
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    ap = np.where(g["n_rel"] > 0, g["ap_sum"] / np.minimum(k, g["n_rel"]).clip(lower=1), 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f_beta(1.0),
        "f2": f_beta(2.0),
        "map": float(ap.mean()),
        "mrr": float(g["rr"].mean()),
    }


def overall_metrics(users, y_true, y_pred, k=10, rel_threshold=4.0, topk_mode="surprise") -> dict[str, float]:
    out = {"mae": mae(y_true, y_pred)}
    out.update(ranking_metrics(users, y_true, y_pred, k, rel_threshold, topk_mode))
    out["fcp"] = fcp(users, y_true, y_pred)
    out["n_test"] = int(len(y_true))
    out["n_test_users"] = int(len(np.unique(users)))
    return out
