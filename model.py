"""Keras models from Section 3 / 4.3 of the paper.

- criteria DNN:  (user, item) -> 7 criteria ratings
- overall DNN:   7 z-scored criteria ratings -> overall rating
- single DNN:    (user, item) -> overall rating (paper's baseline)
"""
from __future__ import annotations

import keras
from keras import layers

INIT_STD = 0.05
CRITERIA_HIDDEN = (128, 64, 32, 16, 8)
OVERALL_HIDDEN = (64, 32, 16, 8)
SINGLE_HIDDEN = (512, 256, 128, 64, 32, 16, 8)

# Paper used Keras 2; for the optimizer sweep we use the Keras 2.2 default learning
# rates for every optimizer except Adam, which the paper set to 0.001 explicitly.
OPTIMIZER_LR = {
    "adam": 0.001,
    "adagrad": 0.01,
    "rmsprop": 0.001,
    "adamax": 0.002,
    "adadelta": 1.0,
    "sgd": 0.01,
}


def make_optimizer(name: str = "adam", lr: float | None = None) -> keras.optimizers.Optimizer:
    name = name.lower()
    lr = OPTIMIZER_LR[name] if lr is None else lr
    cls = {
        "adam": keras.optimizers.Adam,
        "adagrad": keras.optimizers.Adagrad,
        "rmsprop": keras.optimizers.RMSprop,
        "adamax": keras.optimizers.Adamax,
        "adadelta": keras.optimizers.Adadelta,
        "sgd": keras.optimizers.SGD,
    }[name]
    return cls(learning_rate=lr)


def _init():
    # Seeds come from keras.utils.set_random_seed() called before each model is built.
    return keras.initializers.RandomNormal(mean=0.0, stddev=INIT_STD)


def _dense_stack(x, hidden, prefix):
    for k, units in enumerate(hidden):
        x = layers.Dense(units, activation="relu", kernel_initializer=_init(), name=f"{prefix}_dense{k}_{units}")(x)
    return x


def build_embedding_dnn(
    n_users: int,
    n_items: int,
    emb_dim: int,
    hidden: tuple[int, ...],
    n_out: int,
    optimizer: str = "adam",
    name: str = "embedding_dnn",
) -> keras.Model:
    """User/item embeddings -> concat -> ReLU dense stack -> ReLU output."""
    u_in = keras.Input(shape=(1,), dtype="int32", name="user_idx")
    i_in = keras.Input(shape=(1,), dtype="int32", name="item_idx")
    u = layers.Flatten()(layers.Embedding(n_users, emb_dim, embeddings_initializer=_init(), name="user_emb")(u_in))
    i = layers.Flatten()(layers.Embedding(n_items, emb_dim, embeddings_initializer=_init(), name="item_emb")(i_in))
    x = layers.Concatenate()([u, i])
    x = _dense_stack(x, hidden, "h")
    out = layers.Dense(n_out, activation="relu", kernel_initializer=_init(), name="out")(x)
    model = keras.Model([u_in, i_in], out, name=name)
    model.compile(optimizer=make_optimizer(optimizer), loss="mae")
    return model


def build_criteria_dnn(n_users, n_items, emb_dim=64, hidden=CRITERIA_HIDDEN, n_criteria=7, optimizer="adam"):
    return build_embedding_dnn(n_users, n_items, emb_dim, hidden, n_criteria, optimizer, name="criteria_dnn")


def build_single_dnn(n_users, n_items, emb_dim=256, hidden=SINGLE_HIDDEN, optimizer="adam"):
    return build_embedding_dnn(n_users, n_items, emb_dim, hidden, 1, optimizer, name="single_dnn")


def build_overall_dnn(n_criteria=7, hidden=OVERALL_HIDDEN, optimizer="adam") -> keras.Model:
    x_in = keras.Input(shape=(n_criteria,), name="criteria_z")
    x = _dense_stack(x_in, hidden, "h")
    out = layers.Dense(1, activation="relu", kernel_initializer=_init(), name="out")(x)
    model = keras.Model(x_in, out, name="overall_dnn")
    model.compile(optimizer=make_optimizer(optimizer), loss="mae")
    return model
