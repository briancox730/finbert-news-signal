"""Headline text helpers shared by the scoring and labeling stages.

Deduplication is by *normalized* headline text: newswires repost the same story many times
with trivial whitespace/case differences, and scoring each copy would over-weight it in the
per-day mean. Normalizing once, here, keeps that policy in a single place.
"""
from __future__ import annotations

import hashlib


def normalize_headline(headline: str) -> str:
    """Lower-case and collapse runs of whitespace to a single space."""
    return " ".join(str(headline).lower().split())


def headline_key(headline: str) -> str:
    """Stable content hash of the normalized headline (used as a scoring-cache key)."""
    return hashlib.md5(normalize_headline(headline).encode("utf-8")).hexdigest()
