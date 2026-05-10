"""
Embedding-based link prediction using Node2Vec.

Node2Vec learns low-dimensional node representations via biased random
walks on the graph, then scores edges by the similarity of endpoint vectors.
"""

from typing import Dict, List, Tuple

import numpy as np
import networkx as nx


def train_node2vec(
    G: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 200,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    seed: int = 42,
    quiet: bool = True,
) -> Dict:
    """
    Train Node2Vec embeddings on the graph G.

    Uses biased random walks to generate sequences of node "words",
    then trains a Word2Vec model to produce node embeddings.

    Parameters
    ----------
    G          : undirected NetworkX graph
    dimensions : size of each embedding vector
    walk_length: number of steps per random walk
    num_walks  : number of random walks starting from each node
    p          : return parameter — controls revisiting (1 = neutral)
    q          : in-out parameter — < 1 favours BFS (local), > 1 favours DFS
    window     : Word2Vec context window size
    seed       : random seed for reproducibility
    quiet      : suppress verbose output from node2vec

    Returns
    -------
    dict with keys:
      'model'      : trained gensim Word2Vec model
      'embeddings' : dict {node_id: np.ndarray}
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        raise ImportError(
            "node2vec package not found. Install with:  pip install node2vec"
        )

    print(
        f"[node2vec] Training: dim={dimensions}, walks={num_walks}, "
        f"walk_len={walk_length}, p={p}, q={q}"
    )
    nv = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=1,
        seed=seed,
        quiet=quiet,
    )
    model = nv.fit(window=window, min_count=1, batch_words=4, seed=seed)

    # Map node keys back to original integer IDs
    embeddings = {}
    for key in model.wv.index_to_key:
        try:
            node_id = int(key)
        except (ValueError, TypeError):
            node_id = key
        embeddings[node_id] = model.wv[key]

    print(f"[node2vec] Trained {len(embeddings)} embeddings of dim {dimensions}")
    return {"model": model, "embeddings": embeddings}


# ── Edge scoring functions ─────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def dot_product_score(a: np.ndarray, b: np.ndarray) -> float:
    """Raw dot-product similarity."""
    return float(np.dot(a, b))


def hadamard_l1_score(a: np.ndarray, b: np.ndarray) -> float:
    """L1 norm of the element-wise (Hadamard) product."""
    return float(np.sum(np.abs(a * b)))


def score_pairs_node2vec(
    embeddings: Dict,
    pairs: List[Tuple],
    method: str = "cosine",
) -> Dict[Tuple, float]:
    """
    Score edge pairs using pre-trained Node2Vec embeddings.

    Parameters
    ----------
    embeddings : dict mapping node ID → embedding vector
    pairs      : list of (u, v) tuples to score
    method     : 'cosine' | 'dot' | 'hadamard'

    Returns
    -------
    dict mapping (u, v) → float score
    """
    _scorers = {
        "cosine": cosine_similarity,
        "dot": dot_product_score,
        "hadamard": hadamard_l1_score,
    }
    if method not in _scorers:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_scorers)}")

    scorer = _scorers[method]
    scores = {}
    missing = 0
    for u, v in pairs:
        if u in embeddings and v in embeddings:
            scores[(u, v)] = scorer(embeddings[u], embeddings[v])
        else:
            scores[(u, v)] = 0.0
            missing += 1

    if missing:
        print(f"[node2vec] Warning: {missing} pairs had missing embeddings (scored 0)")
    return scores
