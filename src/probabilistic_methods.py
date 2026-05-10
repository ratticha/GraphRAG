"""
Probabilistic ranking methods:
  - PageRank              — global authority-based ranking
  - Personalized PageRank — query-biased graph diffusion

Both use NetworkX's built-in power-iteration PageRank solver.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ── Core PageRank wrappers ────────────────────────────────────────────────────

def run_pagerank(
    G: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1.0e-6,
) -> Dict:
    """
    Compute standard (global) PageRank for every node in G.

    Parameters
    ----------
    G        : NetworkX graph (directed or undirected)
    alpha    : damping factor — probability of following an edge
    max_iter : maximum power-iteration steps
    tol      : convergence tolerance

    Returns
    -------
    dict mapping node → PageRank score (sums to 1.0)
    """
    return nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)


def run_personalized_pagerank(
    G: nx.Graph,
    personalization: Dict,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1.0e-6,
) -> Dict:
    """
    Compute Personalized PageRank seeded from a set of nodes.

    The random surfer teleports back to the seed nodes (with probability
    1 - alpha) rather than to a uniform distribution, biasing the
    stationary distribution toward the query context.

    Parameters
    ----------
    G               : NetworkX graph
    personalization : dict {node: weight} — seed nodes with positive weights
    alpha           : damping factor
    max_iter, tol   : PageRank solver settings

    Returns
    -------
    dict mapping node → PPR score (sums to 1.0)
    """
    total = sum(personalization.values())
    if total <= 0:
        raise ValueError("Personalization weights must sum to a positive value.")

    norm_pers = {k: v / total for k, v in personalization.items()}
    return nx.pagerank(
        G,
        alpha=alpha,
        personalization=norm_pers,
        max_iter=max_iter,
        tol=tol,
    )


# ── Edge scoring for link prediction ─────────────────────────────────────────

def score_pairs_pagerank(
    G: nx.Graph,
    pairs: List[Tuple],
    alpha: float = 0.85,
) -> Dict[Tuple, float]:
    """
    Score (u, v) pairs using the global PageRank of v.

    Candidate v nodes with high global authority receive higher scores.
    PageRank is computed once and reused for all pairs.

    Returns
    -------
    dict mapping (u, v) → score
    """
    print("[pagerank] Computing global PageRank ...")
    pr = run_pagerank(G, alpha=alpha)
    return {(u, v): pr.get(v, 0.0) for u, v in pairs}


def score_pairs_personalized_pagerank(
    G: nx.Graph,
    pairs: List[Tuple],
    alpha: float = 0.85,
) -> Dict[Tuple, float]:
    """
    Score (u, v) pairs using Personalized PageRank seeded at u.

    For each unique source node u, one PPR computation is run and all
    target scores for that u are collected — amortising the cost to
    O(unique_u) PPR runs instead of O(|pairs|).

    Returns
    -------
    dict mapping (u, v) → score
    """
    # Group targets by source node
    u_to_vs: Dict = defaultdict(list)
    for u, v in pairs:
        u_to_vs[u].append(v)

    print(f"[ppr] Running PPR for {len(u_to_vs)} unique source nodes ...")
    scores = {}
    for idx, (u, vs) in enumerate(u_to_vs.items()):
        if u not in G:
            for v in vs:
                scores[(u, v)] = 0.0
            continue
        ppr = run_personalized_pagerank(G, {u: 1.0}, alpha=alpha)
        for v in vs:
            scores[(u, v)] = ppr.get(v, 0.0)

        if (idx + 1) % 100 == 0:
            print(f"[ppr]   processed {idx + 1}/{len(u_to_vs)} source nodes")

    return scores
