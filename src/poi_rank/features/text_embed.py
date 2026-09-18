"""POI text representation (spec.md section 5): embed `name + description + tags`
per POI, then TruncatedSVD to `svd_dim` (64) dimensions.

**Canonical path for `make reproduce` is TF-IDF -> SVD-64, not sentence-transformers**
(see docs/DATA_CARD.md "Text embedding: TF-IDF is the canonical default path"):
`sentence-transformers` is an optional extra (`pyproject.toml`
`[project.optional-dependencies] text`), not installed by a plain `uv sync`, and
downloading a ~90MB model on a fresh clone would blow the <5min/CPU-only/no-paid-API
reproducibility budget. `build_poi_text_embeddings` runs whichever method
`configs/features.yaml`'s `text_embedding.method` names and raises a clear,
actionable `ImportError` if `sentence_transformers` is requested but not installed --
it never silently falls back to TF-IDF at runtime (that would make the two paths
indistinguishable). The fallback is a deliberate *config default*, not a try/except.

**What gets cached to `artifacts/poi_emb.npy`:** the FINAL, L2-normalized 64d
(post-SVD) embedding matrix, not a raw pre-SVD embedding. spec.md's own cache-size
citation ("~1.5k x 384 fp16 ~= 1.1 MB") is anchored to the sentence-transformer
path's fixed-width 384d output; TF-IDF has no such fixed-width raw form (its
dimensionality is vocabulary-sized, not model-fixed) and its fit is sub-second on
this catalog size, so there is no recompute-cost reason to cache anything pre-SVD for
the canonical path. Caching the 64d output instead gives every downstream phase
(semantic candidate channel, traveler taste vectors, models/) a byte-identical
embedding source without re-fitting SVD each run -- the actual property spec.md's
caching requirement protects. Documented in docs/DATA_CARD.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from poi_rank.features.config import TextEmbeddingConfig

FloatArray = npt.NDArray[np.float32]


def _poi_text(name: str, description: str, tags: list[str]) -> str:
    tag_text = " ".join(tags) if len(tags) else ""
    return f"{name} {description} {tag_text}"


def build_poi_corpus(pois_df: pd.DataFrame) -> list[str]:
    """One text document per POI (`name + description + tags`), in `pois_df` row order."""
    return [
        _poi_text(name, description, list(tags))
        for name, description, tags in zip(
            pois_df["name"], pois_df["description"], pois_df["tags"], strict=True
        )
    ]


def _l2_normalize(matrix: npt.NDArray[np.float64]) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized: FloatArray = (matrix / norms).astype(np.float32)
    return normalized


def _pad_to_width(matrix: npt.NDArray[np.float64], width: int) -> npt.NDArray[np.float64]:
    """Zero-pad a `(n, k)` matrix to `(n, width)` when `k < width` -- only triggers on
    degenerate/tiny corpora (e.g. a handful of unit-test rows) where TruncatedSVD
    cannot produce `width` components; keeps the output shape contract uniform."""
    if matrix.shape[1] >= width:
        return matrix
    pad = np.zeros((matrix.shape[0], width - matrix.shape[1]))
    return np.hstack([matrix, pad])


def embed_text_tfidf(
    corpus: list[str], svd_dim: int, max_features: int, ngram_max: int, seed: int
) -> FloatArray:
    """Deterministic fallback: TF-IDF(1, `ngram_max`) -> TruncatedSVD(`svd_dim`) ->
    L2-normalize. Deterministic given a fixed corpus and `seed` (TruncatedSVD's
    randomized solver is seeded via `random_state`; TF-IDF itself has no randomness).

    Exercised directly (not only via an import-error branch) by
    `tests/test_poi_features.py::test_tfidf_fallback_path_produces_valid_embeddings`,
    per spec.md's explicit "fallback path is tested in CI" requirement.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features)
    tfidf = vectorizer.fit_transform(corpus)
    n_components = max(1, min(svd_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(tfidf)
    reduced = _pad_to_width(reduced, svd_dim)
    return _l2_normalize(reduced)


def embed_text_sentence_transformer(
    corpus: list[str], model_name: str, svd_dim: int, seed: int
) -> FloatArray:
    """Optional path: `all-MiniLM-L6-v2` (384d) -> TruncatedSVD(`svd_dim`) ->
    L2-normalize. Requires the optional `sentence-transformers` extra
    (`uv sync --extra text`). Deterministic given a fixed model + fixed input (no
    dropout at inference).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "text_embedding.method='sentence_transformers' requires the optional "
            "'text' extra: `uv sync --extra text`. The canonical default "
            "(text_embedding.method='tfidf') requires no extra install."
        ) from exc

    model = SentenceTransformer(model_name)
    raw = np.asarray(model.encode(corpus, show_progress_bar=False), dtype=np.float64)
    n_components = max(1, min(svd_dim, raw.shape[1] - 1, raw.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(raw)
    reduced = _pad_to_width(reduced, svd_dim)
    return _l2_normalize(reduced)


def compute_poi_text_embeddings(
    pois_df: pd.DataFrame, cfg: TextEmbeddingConfig, seed: int
) -> FloatArray:
    """Compute the POI text embedding matrix fresh (no cache read/write), row-aligned
    to `pois_df`, via whichever method `cfg.method` names."""
    corpus = build_poi_corpus(pois_df)
    if cfg.method == "tfidf":
        return embed_text_tfidf(
            corpus, cfg.svd_dim, cfg.tfidf_max_features, cfg.tfidf_ngram_max, seed
        )
    if cfg.method == "sentence_transformers":
        return embed_text_sentence_transformer(
            corpus, cfg.sentence_transformer_model, cfg.svd_dim, seed
        )
    raise ValueError(f"unknown text_embedding.method: {cfg.method!r}")


def build_poi_text_embeddings(
    pois_df: pd.DataFrame, cfg: TextEmbeddingConfig, seed: int, cache_path: Path
) -> FloatArray:
    """Compute (or load from cache) the POI text embedding matrix, row-aligned to
    `pois_df`.

    Cache semantics: if `cache_path` exists AND its shape matches
    `(len(pois_df), cfg.svd_dim)`, it is loaded and returned as-is -- this is what
    makes `artifacts/poi_emb.npy` a real, committed, reused artifact rather than a
    write-once side effect. Otherwise the configured method is computed fresh and
    written to `cache_path`.
    """
    if cache_path.exists():
        cached: FloatArray = np.load(cache_path)
        if cached.shape == (len(pois_df), cfg.svd_dim):
            return cached.astype(np.float32)

    embeddings = compute_poi_text_embeddings(pois_df, cfg, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings
