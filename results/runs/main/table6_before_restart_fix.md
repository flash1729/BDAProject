# Table 6: paper vs ours

Settings: 5 repeat(s) x 5-fold CV, max 50 epochs, early stopping patience 5 on a 1% validation split, K=10, relevant = true overall >= 4, top-K mode `surprise`. Mean ± sample std over all (repeat, fold) runs.

## Our model (chained criteria DNN -> overall DNN)

| Metric | Paper | Ours | Ours - Paper |
|---|---:|---:|---:|
| MAE | 0.7552 ± 0.0050 | 0.8591 ± 0.3114 | +0.1039 |
| F1 | 0.8769 ± 0.0043 | 0.8337 ± 0.1016 | -0.0432 |
| F2 | 0.9295 ± 0.0014 | 0.8153 ± 0.1252 | -0.1142 |
| FCP | 0.5126 ± 0.0160 | 0.6560 ± 0.0210 | +0.1434 |
| MAP | 0.4643 ± 0.0072 | 0.7768 ± 0.0029 | +0.3125 |
| MRR | 0.7664 ± 0.0039 | 0.7768 ± 0.0029 | +0.0104 |

## All methods, ours (paper in brackets)

| Metric | Ours (DMCCF) | Single DNN | SVD | SVD++ | SlopeOne | Baseline estimates | KNN with baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAE | 0.8591 ± 0.3114 (0.7552) | 1.0539 ± 0.9081 (0.7855) | 0.8059 ± 0.0034 (0.8071) | 0.8220 ± 0.0032 (0.8222) | 0.9151 ± 0.0042 (0.9160) | 0.8097 ± 0.0030 (0.8094) | 0.8098 ± 0.0030 (0.8115) |
| F1 | 0.8337 ± 0.1016 (0.8769) | 0.8200 ± 0.1380 (0.8572) | 0.8381 ± 0.0029 (0.8370) | 0.8398 ± 0.0028 (0.8390) | 0.8667 ± 0.0017 (0.8666) | 0.8394 ± 0.0022 (0.8391) | 0.8393 ± 0.0022 (0.8388) |
| F2 | 0.8153 ± 0.1252 (0.9295) | 0.8034 ± 0.1642 (0.9176) | 0.7970 ± 0.0045 (0.7951) | 0.8007 ± 0.0048 (0.7991) | 0.9251 ± 0.0011 (0.9248) | 0.7990 ± 0.0037 (0.7986) | 0.7989 ± 0.0037 (0.7984) |
| FCP | 0.6560 ± 0.0210 (0.5126) | 0.6181 ± 0.1873 (0.4957) | 0.6817 ± 0.0282 (0.5024) | 0.6798 ± 0.0289 (0.5092) | 0.0505 ± 0.0163 (0.5096) | 0.6923 ± 0.0293 (0.5025) | 0.6904 ± 0.0289 (0.5097) |
| MAP | 0.7768 ± 0.0029 (0.4643) | 0.7768 ± 0.0029 (0.4481) | 0.7770 ± 0.0029 (0.3935) | 0.7770 ± 0.0029 (0.3483) | 0.7758 ± 0.0028 (0.3006) | 0.7770 ± 0.0029 (0.3881) | 0.7770 ± 0.0029 (0.3865) |
| MRR | 0.7768 ± 0.0029 (0.7664) | 0.7768 ± 0.0029 (0.7244) | 0.7770 ± 0.0029 (0.7272) | 0.7770 ± 0.0029 (0.7513) | 0.7758 ± 0.0028 (0.7502) | 0.7770 ± 0.0029 (0.7374) | 0.7770 ± 0.0029 (0.7360) |
| Precision | 0.8881 ± 0.0325 | 0.8865 ± 0.0364 | 0.9170 ± 0.0028 | 0.9141 ± 0.0033 | 0.7842 ± 0.0028 | 0.9166 ± 0.0028 | 0.9166 ± 0.0028 |
| Recall | 0.8050 ± 0.1369 | 0.7947 ± 0.1748 | 0.7718 ± 0.0055 | 0.7766 ± 0.0061 | 0.9685 ± 0.0011 | 0.7742 ± 0.0046 | 0.7741 ± 0.0046 |
| runs | 25 | 25 | 25 | 25 | 25 | 25 | 25 |

## Who wins (ours vs best baseline)

| Metric | Best baseline | Ours | Ours better? |
|---|---|---:|:---:|
| MAE | SVD 0.8059 | 0.8591 | no |
| F1 | SlopeOne 0.8667 | 0.8337 | no |
| F2 | SlopeOne 0.9251 | 0.8153 | no |
| FCP | Baseline estimates 0.6923 | 0.6560 | no |
| MAP | Baseline estimates 0.7770 | 0.7768 | no |
| MRR | Baseline estimates 0.7770 | 0.7768 | no |

## Table 5: criteria DNN per-criterion MAE

| Criterion | Paper | Ours | Ours - Paper |
|---|---:|---:|---:|
| Value | 0.8195 ± 0.0082 | 0.9767 ± 0.6334 | +0.1572 |
| Rooms | 0.7612 ± 0.0082 | 1.1878 ± 1.0717 | +0.4266 |
| Location | 0.6581 ± 0.0077 | 0.8332 ± 0.7292 | +0.1751 |
| Cleanliness | 0.6324 ± 0.0112 | 0.8236 ± 0.7235 | +0.1912 |
| Check in / front desk | 0.7582 ± 0.0143 | 0.7927 ± 0.0104 | +0.0345 |
| Service | 0.7401 ± 0.0103 | 1.0497 ± 0.9307 | +0.3096 |
| Business service | 0.8774 ± 0.0053 | 1.0238 ± 0.5799 | +0.1464 |
| Mean | 0.7495 ± 0.0073 | 0.9554 ± 0.2318 | +0.2059 |

## Diagnostics (ours)

- Overall DNN applied to the *true* test criteria: MAE 0.2692 ± 0.0093 (lower bound for the chained model if net 1 were perfect).
- Best epoch chosen by early stopping: criteria DNN 1.0 (range 1-1), overall DNN 32.3 (range 14-49).

## Notes on definitions

- The paper does not state K, the relevance threshold, or what counts as 'recommended'. See `metrics.py` for the definitions used here; `--topk-mode plain` is the alternative.
- MAP@K and MRR@K: most test users have a single test item, so with eq. 19 as written AP equals RR for them and MAP is close to MRR. The paper's much lower MAP implies an unstated variant.
- FCP counts prediction ties as discordant (eq. 18), so methods that predict a constant for unseen users (e.g. SlopeOne) get a very low FCP.
- KNN with baseline is item-based (`pearson_baseline`); user-based would need a ~40 GB similarity matrix.
- Surprise baselines train on the full 80% training split; the DNNs hold out 1% of it for early stopping.
- Data: dedupe on (user, hotel) removes 33 repeat stays, so we have 81,052 ratings vs the paper's 81,085.
