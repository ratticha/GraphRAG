"""
Universal GraphRAG Retrieval Module.

Given a list of sentences and a natural-language query, this module:
  1. Extracts key terms from sentences and the query.
  2. Builds a heterogeneous graph (query, sentence, term nodes).
  3. Ranks nodes using a graph-based method (default: Personalized PageRank).
  4. Returns the top-K most relevant sentences and terms.

Main API
--------
    from src.graphrag_retriever import retrieve

    results = retrieve(sentences, query, top_k=5, method="ppr")
    for sent, score in results["top_sentences"]:
        print(f"[{score:.4f}] {sent}")
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


# ── Stop-word list ────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "up", "out", "off", "over", "under", "again", "then", "once", "and",
    "or", "but", "if", "while", "as", "that", "this", "it", "its",
    "which", "who", "what", "how", "when", "where", "why", "not", "no",
    "also", "just", "more", "most", "other", "some", "such", "than",
    "too", "very", "so", "each", "both", "few", "many", "only", "own",
    "same", "than", "their", "there", "these", "they", "those", "we",
    "you", "he", "she", "him", "her", "his", "our", "your",
}


# ── Text processing ────────────────────────────────────────────────────────────

def extract_terms(text: str, min_len: int = 3) -> List[str]:
    """
    Tokenise text and return non-stop-word terms of at least min_len chars.

    Returns lowercase strings.
    """
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-']*\b", text)
    return [
        t.lower()
        for t in tokens
        if len(t) >= min_len and t.lower() not in _STOP_WORDS
    ]


# ── Graph construction ────────────────────────────────────────────────────────

def build_retrieval_graph(
    sentences: List[str],
    query: str,
    cooccurrence_window: int = 4,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Build the retrieval graph from a list of sentences and a query.

    Node types (stored in 'ntype' attribute):
      "query"    — single node representing the query
      "sent:<i>" — one node per input sentence
      "term:<t>" — one node per unique key term

    Edges:
      query  ↔ term  if the term appears in the query
      sent   ↔ term  if the term appears in the sentence
      term   ↔ term  co-occurrence within the same sentence (sliding window)

    Parameters
    ----------
    sentences           : list of context sentences
    query               : natural-language query string
    cooccurrence_window : max distance between co-occurring terms

    Returns
    -------
    G        : undirected NetworkX graph
    sent_map : dict {"sent:<i>": i} mapping sentence nodes to sentence indices
    """
    G = nx.Graph()
    G.add_node("query", ntype="query", text=query)
    sent_map: Dict[str, int] = {}

    # Wire query to its terms
    query_terms = set(extract_terms(query))
    for term in query_terms:
        t_node = f"term:{term}"
        if t_node not in G:
            G.add_node(t_node, ntype="term", term=term)
        G.add_edge("query", t_node)

    # Wire sentences to their terms
    for i, sent in enumerate(sentences):
        s_node = f"sent:{i}"
        G.add_node(s_node, ntype="sentence", text=sent, index=i)
        sent_map[s_node] = i

        terms = extract_terms(sent)
        unique_terms = list(dict.fromkeys(terms))  # deduplicated, order-preserving

        for term in unique_terms:
            t_node = f"term:{term}"
            if t_node not in G:
                G.add_node(t_node, ntype="term", term=term)
            G.add_edge(s_node, t_node)

        # Co-occurrence edges within sliding window
        for j in range(len(unique_terms)):
            for k in range(j + 1, min(j + cooccurrence_window, len(unique_terms))):
                if unique_terms[j] != unique_terms[k]:
                    G.add_edge(f"term:{unique_terms[j]}", f"term:{unique_terms[k]}")

    return G, sent_map


# ── Ranking methods ────────────────────────────────────────────────────────────

def _rank_ppr(G: nx.Graph, query_node: str = "query", alpha: float = 0.85) -> Dict[str, float]:
    """Personalized PageRank seeded at the query node and its term neighbours."""
    seed_nodes = [query_node] + list(G.neighbors(query_node))
    seed_nodes = [n for n in seed_nodes if n in G]
    if not seed_nodes:
        return nx.pagerank(G, alpha=alpha)
    pers = {n: 1.0 / len(seed_nodes) for n in seed_nodes}
    return nx.pagerank(G, alpha=alpha, personalization=pers)


def _rank_pagerank(G: nx.Graph, alpha: float = 0.85, **_) -> Dict[str, float]:
    """Standard (global) PageRank."""
    return nx.pagerank(G, alpha=alpha)


def _rank_common_neighbours(G: nx.Graph, query_node: str = "query", **_) -> Dict[str, float]:
    """Score nodes by Common Neighbours with the query node."""
    q_nb = set(G.neighbors(query_node)) if query_node in G else set()
    return {
        n: float(len(q_nb & set(G.neighbors(n))))
        for n in G.nodes()
        if n != query_node
    }


def _rank_jaccard(G: nx.Graph, query_node: str = "query", **_) -> Dict[str, float]:
    """Score nodes by Jaccard Coefficient with the query node."""
    q_nb = set(G.neighbors(query_node)) if query_node in G else set()
    scores = {}
    for n in G.nodes():
        if n == query_node:
            continue
        n_nb = set(G.neighbors(n))
        union = q_nb | n_nb
        scores[n] = len(q_nb & n_nb) / len(union) if union else 0.0
    return scores


def _rank_adamic_adar(G: nx.Graph, query_node: str = "query", **_) -> Dict[str, float]:
    """Score nodes by Adamic-Adar with the query node."""
    q_nb = set(G.neighbors(query_node)) if query_node in G else set()
    scores = {}
    for n in G.nodes():
        if n == query_node:
            continue
        n_nb = set(G.neighbors(n))
        score = sum(
            1.0 / math.log(G.degree(w))
            for w in q_nb & n_nb
            if G.degree(w) > 1
        )
        scores[n] = score
    return scores


_METHODS = {
    "ppr": _rank_ppr,
    "pagerank": _rank_pagerank,
    "common_neighbours": _rank_common_neighbours,
    "jaccard": _rank_jaccard,
    "adamic_adar": _rank_adamic_adar,
}


# ── Main retrieval function ────────────────────────────────────────────────────

def retrieve(
    sentences: List[str],
    query: str,
    top_k: int = 5,
    method: str = "ppr",
    alpha: float = 0.85,
) -> Dict:
    """
    GraphRAG retrieval: rank sentences relevant to a query using graph methods.

    Parameters
    ----------
    sentences : list of context sentences to rank
    query     : natural-language question or search query
    top_k     : number of top results to return
    method    : ranking method — one of:
                  'ppr'              (Personalized PageRank, default)
                  'pagerank'         (global PageRank)
                  'common_neighbours'
                  'jaccard'
                  'adamic_adar'
    alpha     : damping factor used by PageRank variants

    Returns
    -------
    dict with keys:
      'top_sentences' : list of (sentence_text, score) tuples, ranked descending
      'top_terms'     : list of (term_string, score) tuples, ranked descending
      'all_scores'    : dict node_id → score for the full graph
      'graph'         : the constructed NetworkX retrieval graph
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(_METHODS)}"
        )

    G, sent_map = build_retrieval_graph(sentences, query)

    # Run the chosen ranking method
    rank_fn = _METHODS[method]
    scores = rank_fn(G, query_node="query", alpha=alpha)

    # Collect sentence scores
    sent_scores = [
        (sentences[sent_map[s_node]], scores.get(s_node, 0.0))
        for s_node in sent_map
    ]
    sent_scores.sort(key=lambda x: x[1], reverse=True)

    # Collect term scores
    term_scores = [
        (n.replace("term:", ""), scores.get(n, 0.0))
        for n in G.nodes()
        if n.startswith("term:")
    ]
    term_scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "top_sentences": sent_scores[:top_k],
        "top_terms": term_scores[:top_k],
        "all_scores": scores,
        "graph": G,
    }


def retrieve_batch(
    sentence_lists: List[List[str]],
    queries: List[str],
    top_k: int = 5,
    method: str = "ppr",
    alpha: float = 0.85,
) -> List[Dict]:
    """
    Run retrieve() for a batch of (sentences, query) pairs.

    Returns a list of result dicts in the same order as the inputs.
    """
    if len(sentence_lists) != len(queries):
        raise ValueError("sentence_lists and queries must have the same length.")
    return [
        retrieve(sents, q, top_k=top_k, method=method, alpha=alpha)
        for sents, q in zip(sentence_lists, queries)
    ]
