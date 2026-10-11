import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data  # noqa: E402
import metrics  # noqa: E402

SYNTHETIC = (
    "<Author>alice\r\n"
    "<Content>Great stay, would\r return\r\n"
    "<Date>Jan 6, 2009\r\n"
    "<Rating>5\t4\t5\t5\t5\t4\t-1\t4\t\r\n"
    "\r\n"
    "<Author>A TripAdvisor Member\r\n"
    "<Content>Fine hotel\r\n"
    "<Date>Feb 1, 2009\r\n"
    "<Rating>4\t4\t4\t4\t4\t4\t4\t4\t\r\n"
    "\r\n"
    "<Author>bob\r\n"
    "<Content>Clean rooms\rand a stray carriage return in the middle\r\n"
    "<Date>Mar 3, 2009\r\n"
    "<Rating>3\t2\t3\t4\t5\t1\t2\t3\t\r\n"
    "\r\n"
)


def test_parse_and_filter(tmp_path):
    f = tmp_path / "hotel_123_parsed.txt"
    f.write_bytes(SYNTHETIC.encode("utf-8"))

    raw = data.parse_file(f)
    assert [r["user"] for r in raw] == ["alice", "A TripAdvisor Member", "bob"]
    assert all(r["hotel"] == "123" for r in raw)
    assert raw[0]["ratings"] == [5, 4, 5, 5, 5, 4, -1, 4]
    assert raw[2]["ratings"] == [3, 2, 3, 4, 5, 1, 2, 3]
    assert raw[2]["date"] == "Mar 3, 2009"
    assert "\r" not in raw[2]["content"] and "stray carriage return" in raw[2]["content"]

    df = data.filter_reviews(raw)
    assert df["user"].tolist() == ["bob"]  # alice has a -1, the anonymous member is dropped
    assert df[data.RATING_COLS].iloc[0].tolist() == [3, 2, 3, 4, 5, 1, 2, 3]

    ratings, users, items = data.index_ids(df)
    assert ratings.columns.tolist() == ["user_idx", "item_idx", *data.RATING_COLS]
    assert users["user"].tolist() == ["bob"] and items["hotel_id"].tolist() == [123]


def test_fcp_matches_definition():
    # one user, 3 items: pairs (true, pred) give 2 concordant, 1 discordant (a tie in pred)
    users = [0, 0, 0, 1]
    true = [5, 3, 1, 4]
    pred = [4.5, 3.0, 3.0, 2.0]
    assert np.isclose(metrics.fcp(users, true, pred), 2 / 3)


def test_ranking_metrics_simple():
    users = [0, 0, 0]
    true = [5, 2, 4]
    pred = [4.8, 4.5, 3.0]  # ranking: item0 (rel), item1 (not rel), item2 (rel)
    m = metrics.ranking_metrics(users, true, pred, k=10, rel_threshold=4, topk_mode="plain")
    assert np.isclose(m["mrr"], 1.0)
    assert np.isclose(m["map"], (1 / 1 + 2 / 3) / 2)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 1.0)
    s = metrics.ranking_metrics(users, true, pred, k=10, rel_threshold=4, topk_mode="surprise")
    assert np.isclose(s["precision"], 1 / 2) and np.isclose(s["recall"], 1 / 2)
