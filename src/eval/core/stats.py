"""İstatistik: exact McNemar + bootstrap CI + paired delta (saf numpy)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _binom_sf_two_sided(b: int, n: int) -> float:
    """Exact McNemar: P(X<=min(b,n-b) or X>=max(b,n-b)) under Binom(n, 0.5).

    Çift yönlü exact binom testi; n küçükken ki-kare yerine.
    """
    if n == 0:
        return 1.0
    # P(X = k) = C(n,k) / 2^n
    # log-space binomial coeffs
    from math import comb

    # two-sided: sum of probabilities of outcomes as or more extreme than observed
    # observed discordant favoring A: b, favoring B: c, n=b+c
    # under H0, P(X>=b) * 2 with mid-p care — klasik exact:
    k = min(b, n - b)
    # sum P(X <= k) + P(X >= n-k) = 2 * sum_{i=0..k} C(n,i)/2^n  (when k < n/2)
    # when k == n/2, don't double-count the middle
    total = 0.0
    half = 2.0**n
    for i in range(k + 1):
        total += comb(n, i)
    p = 2.0 * total / half
    if n % 2 == 0 and k == n // 2:
        # middle term counted twice
        p -= comb(n, k) / half
    return float(min(1.0, p))


def mcnemar_exact(
    hits_a: Sequence[float] | np.ndarray,
    hits_b: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Hit@1 ikili vektörleri için exact McNemar.

    b = A doğru, B yanlış
    c = A yanlış, B doğru
    """
    a = np.asarray(hits_a, dtype=np.float64).astype(bool)
    b = np.asarray(hits_b, dtype=np.float64).astype(bool)
    if a.shape != b.shape:
        raise ValueError("hits_a ve hits_b aynı uzunlukta olmalı")
    both_ok = int(np.sum(a & b))
    both_fail = int(np.sum(~a & ~b))
    a_only = int(np.sum(a & ~b))  # b in classic notation
    b_only = int(np.sum(~a & b))  # c
    n_discord = a_only + b_only
    p = _binom_sf_two_sided(a_only, n_discord)
    return {
        "n": int(len(a)),
        "both_ok": both_ok,
        "both_fail": both_fail,
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n_discord,
        "p_value": round(p, 6),
    }


def _cluster_ids(
    n: int,
    paraphrase_of: Sequence[str | None] | None,
    gold_questions: Sequence[str] | None,
) -> list[int]:
    """Parafraz çiftlerini aynı kümede tut.

    paraphrase_of = ana soru metni → o ana sorunun indeksiyle aynı cluster.
    """
    clusters = list(range(n))
    if not paraphrase_of or not gold_questions:
        return clusters
    q_to_i = {q: i for i, q in enumerate(gold_questions)}
    for i, para in enumerate(paraphrase_of):
        if para and para in q_to_i:
            clusters[i] = clusters[q_to_i[para]]
    # normalize to contiguous
    mapping: dict[int, int] = {}
    out: list[int] = []
    next_id = 0
    for c in clusters:
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
        out.append(mapping[c])
    return out


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    clusters: Sequence[int] | None = None,
) -> dict[str, float]:
    """Yüzdelik yöntemle %95 (varsayılan) CI; opsiyonel cluster bootstrap."""
    vals = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means = vals[idx].mean(axis=1)
    else:
        cl = np.asarray(clusters, dtype=np.int64)
        unique = np.unique(cl)
        # cluster -> member indices
        members = [np.where(cl == u)[0] for u in unique]
        means = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            chosen = rng.integers(0, len(unique), size=len(unique))
            sample_idx = np.concatenate([members[i] for i in chosen])
            means[b] = vals[sample_idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean": round(float(vals.mean()), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_boot": n_boot,
    }


def paired_bootstrap_delta(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    clusters: Sequence[int] | None = None,
) -> dict[str, float | bool]:
    """A - B farkının CI'si; 0 dışlanıyorsa significant=True."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("values_a ve values_b aynı uzunlukta olmalı")
    delta = a - b
    ci = bootstrap_ci(delta, n_boot=n_boot, seed=seed, alpha=alpha, clusters=clusters)
    excludes_zero = ci["ci_low"] > 0 or ci["ci_high"] < 0
    return {
        "delta_mean": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "excludes_zero": excludes_zero,
        "n_boot": n_boot,
    }


def compare_systems(
    name_a: str,
    per_query_a: list[dict],
    name_b: str,
    per_query_b: list[dict],
    hit_key: str = "hit@1",
    mrr_key: str = "mrr@10",
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """İki sistemin Hit@1 McNemar + MRR paired-delta özeti."""
    hits_a = [q[hit_key] for q in per_query_a]
    hits_b = [q[hit_key] for q in per_query_b]
    mrr_a = [q[mrr_key] for q in per_query_a]
    mrr_b = [q[mrr_key] for q in per_query_b]
    questions = [q.get("question") for q in per_query_a]
    paras = [q.get("paraphrase_of") for q in per_query_a]
    clusters = _cluster_ids(len(per_query_a), paras, questions)

    mc = mcnemar_exact(hits_a, hits_b)
    delta = paired_bootstrap_delta(
        mrr_a, mrr_b, n_boot=n_boot, seed=seed, clusters=clusters
    )
    hit_ci_a = bootstrap_ci(hits_a, n_boot=n_boot, seed=seed, clusters=clusters)
    hit_ci_b = bootstrap_ci(hits_b, n_boot=n_boot, seed=seed, clusters=clusters)
    mrr_ci_a = bootstrap_ci(mrr_a, n_boot=n_boot, seed=seed, clusters=clusters)
    mrr_ci_b = bootstrap_ci(mrr_b, n_boot=n_boot, seed=seed, clusters=clusters)

    winner = None
    if mc["p_value"] < 0.05 or delta["excludes_zero"]:
        if mrr_ci_a["mean"] >= mrr_ci_b["mean"] and hits_a >= hits_b:
            # prefer mean hit then mrr
            pass
        mean_hit_a = float(np.mean(hits_a))
        mean_hit_b = float(np.mean(hits_b))
        if mean_hit_a > mean_hit_b or (
            mean_hit_a == mean_hit_b and mrr_ci_a["mean"] > mrr_ci_b["mean"]
        ):
            winner = name_a
        elif mean_hit_b > mean_hit_a or (
            mean_hit_a == mean_hit_b and mrr_ci_b["mean"] > mrr_ci_a["mean"]
        ):
            winner = name_b

    return {
        "a": name_a,
        "b": name_b,
        "mcnemar": mc,
        "mrr_delta": delta,
        "hit_ci_a": hit_ci_a,
        "hit_ci_b": hit_ci_b,
        "mrr_ci_a": mrr_ci_a,
        "mrr_ci_b": mrr_ci_b,
        "significant": bool(mc["p_value"] < 0.05 or delta["excludes_zero"]),
        "winner": winner,
    }
