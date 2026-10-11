"""Numbers reported in Nassar, Jafar, Rahhal (KBS 187, 2020), used for comparisons."""

PAPER_TABLE1 = {"users": 72119, "hotels": 1850, "ratings": 81085, "sparsity": 0.9994}

# counts of ratings 1..5
PAPER_TABLE2 = {
    "value": [4268, 5853, 10374, 24696, 35894],
    "rooms": [4002, 5517, 10493, 25019, 36054],
    "location": [1464, 3141, 8503, 22177, 45800],
    "cleanliness": [2591, 3585, 8232, 19951, 46726],
    "checkin": [3775, 4468, 11083, 19201, 42558],
    "service": [4086, 4462, 10036, 20114, 42387],
    "business": [4525, 5849, 19624, 21661, 29426],
    "overall": [4246, 6052, 8078, 24180, 38529],
}

# Table 3: chained-model MAE per optimizer (mean, std)
PAPER_TABLE3 = {
    "adadelta": (0.9112, 0.0170),
    "adagrad": (0.7565, 0.0216),
    "adam": (0.7552, 0.0050),
    "adamax": (0.7919, 0.0112),
    "rmsprop": (0.7612, 0.0059),
    "sgd": (3.2805, 0.2129),
}

# Table 4: chained-model MAE per hidden-layer config of the criteria DNN
PAPER_TABLE4 = {
    "8": (3.4703, 0.0708),
    "16-8": (3.0678, 0.5612),
    "32-16-8": (0.9481, 0.0636),
    "64-32-16-8": (0.7776, 0.0146),
    "128-64-32-16-8": (0.7552, 0.0050),
    "256-128-64-32-16-8": (0.7830, 0.0629),
}

# Table 5: per-criterion MAE of the criteria DNN (mean, std)
PAPER_TABLE5 = {
    "value": (0.8195, 0.0082),
    "rooms": (0.7612, 0.0082),
    "location": (0.6581, 0.0077),
    "cleanliness": (0.6324, 0.0112),
    "checkin": (0.7582, 0.0143),
    "service": (0.7401, 0.0103),
    "business": (0.8774, 0.0053),
    "mean": (0.7495, 0.0073),
}

# Table 6: overall-rating metrics (mean, std)
PAPER_TABLE6 = {
    "svd": {"mae": (0.8071, 0.0043), "f1": (0.8370, 0.0035), "f2": (0.7951, 0.0031),
            "fcp": (0.5024, 0.0210), "map": (0.3935, 0.0057), "mrr": (0.7272, 0.0054)},
    "svdpp": {"mae": (0.8222, 0.0071), "f1": (0.8390, 0.0035), "f2": (0.7991, 0.0057),
              "fcp": (0.5092, 0.0056), "map": (0.3483, 0.0049), "mrr": (0.7513, 0.0018)},
    "slopeone": {"mae": (0.9160, 0.0053), "f1": (0.8666, 0.0018), "f2": (0.9248, 0.0011),
                 "fcp": (0.5096, 0.0661), "map": (0.3006, 0.0026), "mrr": (0.7502, 0.0037)},
    "baselineonly": {"mae": (0.8094, 0.0019), "f1": (0.8391, 0.0020), "f2": (0.7986, 0.0029),
                     "fcp": (0.5025, 0.0117), "map": (0.3881, 0.0023), "mrr": (0.7374, 0.0028)},
    "knn": {"mae": (0.8115, 0.0041), "f1": (0.8388, 0.0019), "f2": (0.7984, 0.0031),
            "fcp": (0.5097, 0.0127), "map": (0.3865, 0.0041), "mrr": (0.7360, 0.0010)},
    "single_dnn": {"mae": (0.7855, 0.0153), "f1": (0.8572, 0.0024), "f2": (0.9176, 0.0057),
                   "fcp": (0.4957, 0.0102), "map": (0.4481, 0.0091), "mrr": (0.7244, 0.0092)},
    "ours": {"mae": (0.7552, 0.0050), "f1": (0.8769, 0.0043), "f2": (0.9295, 0.0014),
             "fcp": (0.5126, 0.0160), "map": (0.4643, 0.0072), "mrr": (0.7664, 0.0039)},
}

METHOD_LABELS = {
    "ours": "Ours (DMCCF)",
    "single_dnn": "Single DNN",
    "svd": "SVD",
    "svdpp": "SVD++",
    "slopeone": "SlopeOne",
    "baselineonly": "Baseline estimates",
    "knn": "KNN with baseline",
}
