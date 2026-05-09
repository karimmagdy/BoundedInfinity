#!/usr/bin/env python3
"""
score_text_pairs.py — Offline computation of stronger semantic metrics on logged
(original, reconstructed) agent text pairs.

Reads result JSONs produced by run_llm_50agents.py (after the harness was modified
to set log_text_pairs=True), and computes per-pair and aggregate scores using:

  (a) sentence-embedding cosine similarity (default: sentence-transformers/all-MiniLM-L6-v2;
      falls back to BGE if requested)
  (b) BERTScore F1 (Zhang et al., ICLR 2020)

Both replace the surface-level Jaccard / TF-IDF cosine in the existing pipeline.

Usage:
    python experiments/score_text_pairs.py \
        --input experiments/results/experiment_6_llm_50agents_relogged.json \
        --output experiments/results/experiment_6_semantic_scored.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def load_pairs(path: Path) -> list[dict[str, Any]]:
    """Extract every text pair from a result file (or list of result files).

    The result schema written by RunMetrics.to_dict has shape::

        {
          "backend": "...",
          "config": {...},
          ...
          "text_pairs": [
              {"agent_id": ..., "original": ..., "reconstructed": ...,
               "category": ..., "jaccard": ..., "tfidf_cosine": ...},
              ...
          ]
        }

    Top-level result files may be a single dict, a list of dicts, or a dict
    with nested "runs" or "results" lists. We cover all three.
    """
    raw = json.loads(path.read_text())
    runs: list[dict[str, Any]] = []
    if isinstance(raw, list):
        runs = raw
    elif isinstance(raw, dict):
        if "text_pairs" in raw:
            runs = [raw]
        elif "runs" in raw:
            runs = raw["runs"]
        elif "results" in raw:
            runs = raw["results"]
        else:
            # try every nested list
            for v in raw.values():
                if isinstance(v, list):
                    runs.extend(x for x in v if isinstance(x, dict))

    pairs: list[dict[str, Any]] = []
    for r in runs:
        for p in r.get("text_pairs", []):
            pairs.append({
                "backend": r.get("backend"),
                "cache_size": r.get("config", {}).get("cache_size"),
                "seed": r.get("config", {}).get("seed"),
                **p,
            })
    return pairs


def _try_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] sentence-transformers unavailable ({exc}); skipping embedding cosine")
        return None


def _try_bertscore():
    try:
        import bert_score
        return bert_score
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] bert-score unavailable ({exc}); skipping BERTScore")
        return None


def score(pairs: list[dict[str, Any]],
          sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[dict[str, Any]]:
    """Augment each pair with embedding_cosine and bertscore_f1, when libs are available."""
    if not pairs:
        return pairs

    originals = [p["original"] for p in pairs]
    reconstructed = [p["reconstructed"] for p in pairs]

    # --- embedding cosine ---
    sbert = _try_sbert(sbert_model)
    if sbert is not None:
        import numpy as np  # local import so script can be inspected without numpy
        embeds_o = sbert.encode(originals, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        embeds_r = sbert.encode(reconstructed, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        cosines = (embeds_o * embeds_r).sum(axis=1)
        for p, c in zip(pairs, cosines):
            p["embedding_cosine"] = float(round(float(c), 4))

    # --- BERTScore ---
    bs = _try_bertscore()
    if bs is not None:
        # use roberta-large by default; F1 reported
        P, R, F1 = bs.score(reconstructed, originals, lang="en", rescale_with_baseline=True, verbose=False)
        for p, f1 in zip(pairs, F1.tolist()):
            p["bertscore_f1"] = round(float(f1), 4)

    return pairs


def aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by (backend, cache_size) and report mean + std for every available metric."""
    out: dict[str, Any] = {}
    seen_metrics: list[str] = []
    for m in ["jaccard", "tfidf_cosine", "embedding_cosine", "bertscore_f1"]:
        if any(m in p for p in pairs):
            seen_metrics.append(m)

    by_group: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for p in pairs:
        k = (p.get("backend"), p.get("cache_size"))
        by_group.setdefault(k, []).append(p)

    for (backend, cache_size), group in by_group.items():
        key = f"{backend}_M{cache_size}"
        out[key] = {"n_pairs": len(group)}
        for metric in seen_metrics:
            vals = [p[metric] for p in group if metric in p]
            if not vals:
                continue
            out[key][metric] = {
                "mean": round(mean(vals), 4),
                "std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to result JSON with text_pairs")
    ap.add_argument("--output", required=True, help="Path to write augmented + aggregated JSON")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    inp = Path(args.input)
    pairs = load_pairs(inp)
    print(f"[info] loaded {len(pairs)} text pairs from {inp}")
    if not pairs:
        print("[warn] no text_pairs found; ensure run_llm_50agents.py was rerun with the modified harness")
        return

    scored = score(pairs, sbert_model=args.sbert_model)
    aggregated = aggregate(scored)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_pairs": len(scored),
        "metrics": sorted({k for p in scored for k in p if k in {"jaccard", "tfidf_cosine", "embedding_cosine", "bertscore_f1"}}),
        "per_group": aggregated,
        "pairs": scored,
    }, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
