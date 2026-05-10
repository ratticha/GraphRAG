"""
Heuristic link-prediction methods:
  - Common Neighbours
  - Jaccard Coefficient
  - Adamic-Adar Index
  - Katz Index (matrix-power approximation)

All functions share the same interface:
    score_pairs(G, pairs) -> Dict[Tuple, float]
"""

import math
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp


# ── Individual scoring functions ──────────────────────────────────────────────

def common_neighbours(
    G: nx.Graph, pairs: List[Tuple]
) -> Dict[Tuple, float]:
    """
    Common Neighbours score.

    score(u, v) = |N(u) ∩ N(v)|

    Intuition: nodes that share many neighbours are likely connected.
    """
    scores = {}
    for u, v in pairs:
        nu = set(G.neighbors(u)) if u in G else set()
        nv = set(G.neighbors(v)) if v in G else set()
        scores[(u, v)] = float(len(nu & nv))
    return scores


def jaccard_coefficient(
    G: nx.Graph, pairs: List[Tuple]
) -> Dict[Tuple, float]:
    """
    Jaccard Coefficient.

    score(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    Normalises Common Neighbours by total neighbourhood size,
    penalising high-degree hubs.
    """
    scores = {}
    for u, v in pairs:
        nu = set(G.neighbors(u)) if u in G else set()
        nv = set(G.neighbors(v)) if v in G else set()
        union = nu | nv
        scores[(u, v)] = len(nu & nv) / len(union) if union else 0.0
    return scores


def adamic_adar(
    G: nx.Graph, pairs: List[Tuple]
) -> Dict[Tuple, float]:
    """
    Adamic-Adar Index.

    score(u, v) = Σ_{w ∈ N(u)∩N(v)}  1 / log(|N(w)|)

    Gives higher weight to shared neighbours with low degree
    (rare co-occurrences are more informative).
    """
    scores = {}
    for u, v in pairs:
        nu = set(G.neighbors(u)) if u in G else set()
        nv = set(G.neighbors(v)) if v in G else set()
        score = 0.0
        for w in nu & nv:
            deg = G.degree(w)
            if deg > 1:
                score += 1.0 / math.log(deg)
        scores[(u, v)] = score
    return scores


def katz_index(
    G: nx.Graph,
    pairs: List[Tuple],
    beta: float = 0.005,
    max_power: int = 5,
) -> Dict[Tuple, float]:
    """
    Katz Index (truncated series approximation).

    score(u, v) = Σ_{l=1}^{max_power} β^l * (A^l)[u, v]

    Counts paths of all lengths between u and v, down-weighted
    exponentially by length β. Uses sparse matrix powers for efficiency.

    Parameters
    ----------
    beta      : decay factor per path length (must be < 1/λ_max)
    max_power : truncation depth of the path-counting series
    """
    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    rows, cols = [], []
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            rows += [i, j]
            cols += [j, i]

    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float32
    )

    katz = sp.csr_matrix((n, n), dtype=np.float32)
    A_pow = A.copy()
    for l in range(1, max_power + 1):
        katz = katz + (beta ** l) * A_pow
        A_pow = A_pow.dot(A)

    scores = {}
    for u, v in pairs:
        if u in node_idx and v in node_idx:
            scores[(u, v)] = float(katz[node_idx[u], node_idx[v]])
        else:
            scores[(u, v)] = 0.0
    return scores


# ── Batch runner ──────────────────────────────────────────────────────────────

def score_pairs_heuristics(
    G: nx.Graph,
    pairs: List[Tuple],
    beta: float = 0.005,
    katz_depth: int = 5,
) -> Dict[str, Dict[Tuple, float]]:
    """
    Run all four heuristic methods on a list of (u, v) pairs.

    Returns
    -------
    dict mapping method name → score dict
    """
    print("[heuristics] Common Neighbours ...")
    cn = common_neighbours(G, pairs)

    print("[heuristics] Jaccard Coefficient ...")
    jc = jaccard_coefficient(G, pairs)

    print("[heuristics] Adamic-Adar ...")
    aa = adamic_adar(G, pairs)

    print("[heuristics] Katz Index ...")
    kz = katz_index(G, pairs, beta=beta, max_power=katz_depth)

    return {
        "common_neighbours": cn,
        "jaccard": jc,
        "adamic_adar": aa,
        "katz": kz,
    }
