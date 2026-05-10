"""
Data loading utilities for:
  - Wikipedia Vote Network (SNAP)
  - HotpotQA distractor development set
"""

import os
import gzip
import json
import random
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx


# ── Download helpers ──────────────────────────────────────────────────────────

WIKI_VOTE_URL = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
HOTPOTQA_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)


def download_wiki_vote(raw_dir: str = "data/raw") -> str:
    """
    Download and extract the Wikipedia Vote Network from SNAP.

    Returns the path to the extracted .txt file.
    Skips download if the file already exists.
    """
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    gz_path = os.path.join(raw_dir, "wiki-Vote.txt.gz")
    txt_path = os.path.join(raw_dir, "wiki-Vote.txt")

    if os.path.exists(txt_path):
        print(f"[data_loader] Found existing file: {txt_path}")
        return txt_path

    if not os.path.exists(gz_path):
        print(f"[data_loader] Downloading Wikipedia Vote Network ...")
        urllib.request.urlretrieve(WIKI_VOTE_URL, gz_path)
        print(f"[data_loader] Download complete: {gz_path}")

    print("[data_loader] Extracting ...")
    with gzip.open(gz_path, "rb") as fin, open(txt_path, "wb") as fout:
        fout.write(fin.read())
    print(f"[data_loader] Extracted to: {txt_path}")
    return txt_path


def download_hotpotqa(raw_dir: str = "data/raw") -> str:
    """
    Download HotpotQA distractor dev set JSON.

    Returns the path to the downloaded file.
    Skips download if the file already exists.
    """
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(raw_dir, "hotpot_dev_distractor_v1.json")
    if os.path.exists(path):
        print(f"[data_loader] Found existing file: {path}")
        return path

    print("[data_loader] Downloading HotpotQA distractor dev set ...")
    urllib.request.urlretrieve(HOTPOTQA_URL, path)
    print(f"[data_loader] Saved to: {path}")
    return path


# ── Graph loaders ─────────────────────────────────────────────────────────────

def load_wiki_vote_graph(txt_path: str) -> nx.DiGraph:
    """
    Parse the Wikipedia Vote .txt file into a directed NetworkX graph.

    Lines beginning with '#' are comments and are skipped.
    Each data line has the format: <FromNodeId>\\t<ToNodeId>
    """
    G = nx.DiGraph()
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                src, dst = int(parts[0]), int(parts[1])
                G.add_edge(src, dst)

    print(
        f"[data_loader] Wiki-Vote graph loaded: "
        f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
    )
    return G


def load_hotpotqa(json_path: str, max_samples: Optional[int] = None) -> List[dict]:
    """
    Load HotpotQA distractor JSON and return a list of question records.

    Each record is a dict with keys: _id, question, answer, context,
    supporting_facts, type, level.

    Parameters
    ----------
    json_path   : path to hotpot_dev_distractor_v1.json
    max_samples : if given, only the first N records are returned
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"[data_loader] Loaded {len(data):,} HotpotQA records")
    return data


# ── Train / test split ────────────────────────────────────────────────────────

def split_edges(
    G: nx.Graph,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[nx.Graph, List[tuple], List[tuple]]:
    """
    Randomly split edges into train (80 %) and test (20 %) sets.

    Negative test edges are sampled uniformly from non-existing pairs
    to match the number of positive test edges.

    Parameters
    ----------
    G          : original graph (directed or undirected)
    test_ratio : fraction of edges held out for testing
    seed       : random seed

    Returns
    -------
    G_train  : subgraph containing only training edges (same type as G)
    test_pos : list of positive (true) test edges
    test_neg : list of sampled negative (non-edge) test pairs
    """
    random.seed(seed)
    edges = list(G.edges())
    random.shuffle(edges)

    n_test = int(len(edges) * test_ratio)
    test_pos = edges[:n_test]
    train_edges = edges[n_test:]

    # Rebuild training graph preserving node set
    G_train = G.__class__()
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(train_edges)

    # Sample negatives: random non-edges of the original graph
    nodes = list(G.nodes())
    existing_edges = set(G.edges())
    neg_set: set = set()
    max_tries = n_test * 20

    for _ in range(max_tries):
        if len(neg_set) >= n_test:
            break
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and (u, v) not in existing_edges and (u, v) not in neg_set:
            neg_set.add((u, v))

    test_neg = list(neg_set)[:n_test]

    print(
        f"[data_loader] Split complete — "
        f"train: {len(train_edges):,} edges, "
        f"test pos: {len(test_pos):,}, "
        f"test neg: {len(test_neg):,}"
    )
    return G_train, test_pos, test_neg
