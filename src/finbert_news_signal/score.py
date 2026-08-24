"""Scoring stage: headline -> sentiment score in [-1, 1].

Two scorers, one contract (higher = more positive):

* ``lexicon`` (default) -- a finance polarity word list, ``(pos - neg) / (pos + neg + 1)``.
  Zero heavy dependencies; this is the path CI and the offline quickstart exercise.
* ``finbert`` (optional) -- ``P(positive) - P(negative)`` from ProsusAI/finbert. Requires the
  ``[finbert]`` extra (transformers + torch). Imported lazily so the package installs and
  runs without it.

Both scorers are point-in-time safe: they read only the headline text, never any outcome or
future information. Scores are deduplicated by normalized headline and cached by content hash
so each distinct headline is scored exactly once.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache as cache_mod
from .text import headline_key, normalize_headline

# A compact finance polarity lexicon (Loughran-McDonald flavored). Deliberately small and
# auditable -- it is the always-available fallback, not a claim to state-of-the-art accuracy.
POSITIVE_WORDS = frozenset({
    "beat", "beats", "surge", "surged", "soar", "soars", "soared", "jump", "jumps",
    "rise", "rises", "rose", "gain", "gains", "gained", "record", "raises", "raised",
    "upgrade", "upgraded", "outperform", "buy", "bullish", "strong", "growth", "profit",
    "profits", "boost", "boosts", "wins", "win", "approval", "approved", "rally",
    "rallies", "top", "tops", "topped", "high", "higher", "positive", "expands",
    "expansion", "dividend", "buyback", "upside", "optimistic", "advance", "advances",
})
NEGATIVE_WORDS = frozenset({
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "fall", "falls",
    "fell", "drop", "drops", "dropped", "loss", "losses", "cut", "cuts", "slump",
    "downgrade", "downgraded", "underperform", "sell", "bearish", "weak", "warning",
    "warns", "warn", "probe", "lawsuit", "fraud", "recall", "recalls", "decline",
    "declines", "sink", "sinks", "slashed", "slash", "concern", "concerns", "fears",
    "risk", "risks", "low", "lower", "negative", "layoffs", "bankruptcy", "halt",
    "halts", "delay", "delays", "investigation", "subpoena", "default", "downside",
})

FinbertFn = Callable[[Sequence[str]], np.ndarray]


def score_lexicon(texts: Sequence[str]) -> np.ndarray:
    """Signed polarity in ~[-1, 1] via ``(pos - neg) / (pos + neg + 1)`` on lexicon hits."""
    out = np.empty(len(texts), dtype=float)
    for i, text in enumerate(texts):
        words = normalize_headline(text).replace(",", " ").replace(".", " ").split()
        pos = sum(w in POSITIVE_WORDS for w in words)
        neg = sum(w in NEGATIVE_WORDS for w in words)
        out[i] = (pos - neg) / (pos + neg + 1.0)
    return out


def score_finbert(texts: Sequence[str], *, batch: int = 64, max_length: int = 64) -> np.ndarray:
    """``P(positive) - P(negative)`` from ProsusAI/finbert. Requires the ``[finbert]`` extra.

    transformers/torch are imported here (not at module load) so the package is usable with
    only the lexicon path installed.
    """
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "FinBERT scoring needs the optional extra: pip install 'finbert-news-signal[finbert]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    id2label = model.config.id2label
    pos_idx = next(i for i, label in id2label.items() if label.lower() == "positive")
    neg_idx = next(i for i, label in id2label.items() if label.lower() == "negative")

    out = np.empty(len(texts), dtype=float)
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = [t if t else "." for t in texts[start:start + batch]]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            probs = functional.softmax(model(**enc).logits, dim=1).numpy()
            out[start:start + batch] = probs[:, pos_idx] - probs[:, neg_idx]
    return out


def score_texts(
    texts: Sequence[str], *, scorer: str = "lexicon", finbert_fn: FinbertFn | None = None
) -> np.ndarray:
    """Score a list of texts with the named scorer (no caching, no dedup).

    ``finbert_fn`` lets callers (and tests) inject the FinBERT implementation so the model is
    never downloaded implicitly.
    """
    if scorer == "lexicon":
        return score_lexicon(texts)
    if scorer == "finbert":
        fn = finbert_fn or score_finbert
        return np.asarray(fn(texts), dtype=float)
    raise ValueError(f"unknown scorer {scorer!r} (expected 'lexicon' or 'finbert')")


def score_items(
    items: pd.DataFrame,
    *,
    scorer: str = "lexicon",
    cache_dir: Path | None = None,
    finbert_fn: FinbertFn | None = None,
) -> pd.Series:
    """Score every headline in ``items``, returning a float Series aligned to ``items.index``.

    Each *distinct* normalized headline is scored once. If ``cache_dir`` is given, results are
    written to (and re-read from) a per-scorer scores cache so repeated runs are cheap.

    Args:
        items: long table with at least ``headline`` and ``hnorm`` columns (see ``label``).
        scorer: ``"lexicon"`` or ``"finbert"``.
        cache_dir: optional Parquet scores cache location.
        finbert_fn: optional injected FinBERT scorer (used only when ``scorer='finbert'``).
    """
    if items.empty:
        return pd.Series(dtype=float, index=items.index)

    uniq = items.drop_duplicates(subset="hnorm")[["hnorm", "headline"]].copy()
    uniq["hkey"] = uniq["headline"].map(headline_key)

    have = cache_mod.read_scores(cache_dir, scorer) if cache_dir is not None else None
    known = set(have["hkey"]) if have is not None else set()
    todo = uniq[~uniq["hkey"].isin(known)]

    if len(todo):
        new_scores = score_texts(
            todo["headline"].tolist(), scorer=scorer, finbert_fn=finbert_fn
        )
        new = pd.DataFrame({"hkey": todo["hkey"].to_numpy(), "score": new_scores})
        if have is not None:
            have = pd.concat([have, new], ignore_index=True).drop_duplicates(subset="hkey")
            cache_mod.write_scores(have, cache_dir, scorer)
        else:
            have = new

    score_by_key = dict(zip(have["hkey"], have["score"], strict=False))
    keys = items["headline"].map(headline_key)
    return keys.map(score_by_key).astype(float)
