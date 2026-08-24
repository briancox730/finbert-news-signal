"""Scoring: lexicon sign + monotonicity, and the FinBERT wrapper behind a mock."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from finbert_news_signal.score import score_items, score_lexicon, score_texts


def test_lexicon_sign():
    pos = score_lexicon(["Acme beats estimates as profit surges to a record"])[0]
    neg = score_lexicon(["Acme misses estimates and plunges on a fraud probe"])[0]
    neutral = score_lexicon(["Acme to present at an industry conference"])[0]
    assert pos > 0
    assert neg < 0
    assert neutral == 0.0


def test_lexicon_monotonicity():
    # Adding positive words raises the score; adding negatives lowers it.
    s1 = score_lexicon(["beat"])[0]
    s2 = score_lexicon(["beat surge gain rally"])[0]
    s3 = score_lexicon(["beat surge gain rally miss"])[0]
    assert s2 > s1
    assert s3 < s2
    # Full ordering: all-positive > mixed > all-negative.
    all_pos = score_lexicon(["surge gain rally profit"])[0]
    mixed = score_lexicon(["surge gain miss loss"])[0]
    all_neg = score_lexicon(["plunge loss fraud lawsuit"])[0]
    assert all_pos > mixed > all_neg


def test_lexicon_bounded():
    scores = score_lexicon([
        "surge gain rally profit beat boost",
        "plunge loss fraud lawsuit miss cut",
        "",
    ])
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)


def test_score_module_import_has_no_torch():
    # Importing the scorer must not pull the heavy transformer stack.
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_finbert_wrapper_behind_mock():
    calls = {}

    def fake_finbert(texts):
        calls["texts"] = list(texts)
        return np.array([0.5, -0.25])

    out = score_texts(["good news", "bad news"], scorer="finbert", finbert_fn=fake_finbert)
    assert list(out) == [0.5, -0.25]
    assert calls["texts"] == ["good news", "bad news"]


def test_score_items_routes_finbert_and_caches(tmp_path):
    items = pd.DataFrame({
        "headline": ["alpha up", "beta down", "alpha up"],  # duplicate headline
        "hnorm": ["alpha up", "beta down", "alpha up"],
    })
    call_count = {"n": 0}

    def fake_finbert(texts):
        call_count["n"] += len(texts)
        return np.array([0.9 if "up" in t else -0.9 for t in texts])

    scores = score_items(items, scorer="finbert", cache_dir=tmp_path, finbert_fn=fake_finbert)
    # Aligned to items.index, duplicate scored once.
    assert list(scores) == [0.9, -0.9, 0.9]
    assert call_count["n"] == 2  # only 2 unique headlines hit the model

    # Second call reads from cache; the model is not invoked again.
    call_count["n"] = 0
    scores2 = score_items(items, scorer="finbert", cache_dir=tmp_path, finbert_fn=fake_finbert)
    assert list(scores2) == [0.9, -0.9, 0.9]
    assert call_count["n"] == 0
