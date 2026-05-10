"""
Graph construction utilities.

  build_undirected()      — convert directed graph for heuristic methods
  build_hotpotqa_graph()  — entity/sentence co-occurrence graph from HotpotQA
  extract_entities()      — lightweight capitalisation-based NER
"""

import re
from typing import List, Set, Tuple

import networkx as nx


# ── Wikipedia Vote helpers ────────────────────────────────────────────────────

def build_undirected(G_directed: nx.DiGraph) -> nx.Graph:
    """
    Convert a directed graph to undirected.

    Heuristic methods (Common Neighbours, Jaccard, Adamic-Adar, Katz)
    require an undirected graph because they rely on shared neighbourhoods.
    """
    return G_directed.to_undirected()


# ── HotpotQA graph construction ───────────────────────────────────────────────

def extract_entities(text: str) -> Set[str]:
    """
    Extract candidate entity phrases using capitalisation heuristics.

    Identifies runs of Title-Cased words (≥ 2 chars each) as potential
    named entities, then adds unigram and bigram forms.
    Returns lowercase entity strings longer than 3 characters.
    """
    tokens = re.findall(r"\b[A-Z][a-z]{1,}\b", text)
    entities: Set[str] = set(tokens)
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    entities.update(bigrams)
    return {e.lower() for e in entities if len(e) > 3}


def build_hotpotqa_graph(
    records: List[dict],
    max_records: int = 500,
) -> nx.Graph:
    """
    Build a heterogeneous co-occurrence graph from HotpotQA records.

    Node types (stored in the 'ntype' attribute):
      "query"    — one node per HotpotQA question
      "doc"      — one node per Wikipedia article title
      "sentence" — one node per sentence within a document
      "entity"   — one node per extracted named entity

    Edges represent:
      query    ↔ entity    (entity appears in the question)
      doc      ↔ sentence  (sentence belongs to document)
      sentence ↔ entity    (entity appears in sentence)
      entity   ↔ entity    (co-occurrence within the same sentence)

    Parameters
    ----------
    records     : list of HotpotQA dicts (from data_loader.load_hotpotqa)
    max_records : cap on the number of records to process

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()
    records = records[:max_records]

    for rec in records:
        q_id = rec["_id"]
        q_text = rec.get("question", "")
        q_node = f"query:{q_id}"
        G.add_node(q_node, ntype="query", text=q_text)

        # Add document and sentence nodes
        for doc_title, sentences in rec.get("context", []):
            doc_node = f"doc:{doc_title}"
            G.add_node(doc_node, ntype="doc", title=doc_title)

            for s_idx, sentence in enumerate(sentences):
                sent_node = f"sent:{doc_title}:{s_idx}"
                G.add_node(sent_node, ntype="sentence", text=sentence,
                           doc=doc_title, sent_idx=s_idx)
                G.add_edge(doc_node, sent_node)

                ents = extract_entities(sentence)
                ent_list = list(ents)
                for ent in ent_list:
                    ent_node = f"ent:{ent}"
                    G.add_node(ent_node, ntype="entity", name=ent)
                    G.add_edge(sent_node, ent_node)

                # Co-occurrence: entity pairs within the same sentence
                for i in range(len(ent_list)):
                    for j in range(i + 1, min(i + 5, len(ent_list))):
                        G.add_edge(f"ent:{ent_list[i]}", f"ent:{ent_list[j]}")

        # Connect query node to its entities
        for ent in extract_entities(q_text):
            ent_node = f"ent:{ent}"
            if ent_node in G:
                G.add_edge(q_node, ent_node)

    print(
        f"[graph_builder] HotpotQA graph: "
        f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
        f"from {len(records)} records"
    )
    return G


def get_nodes_by_type(G: nx.Graph, ntype: str) -> List[str]:
    """Return all node IDs whose 'ntype' attribute equals ntype."""
    return [n for n, d in G.nodes(data=True) if d.get("ntype") == ntype]


def get_sentence_nodes(G: nx.Graph) -> List[str]:
    """Convenience wrapper: return all sentence node IDs."""
    return get_nodes_by_type(G, "sentence")


def get_doc_nodes(G: nx.Graph) -> List[str]:
    """Convenience wrapper: return all document node IDs."""
    return get_nodes_by_type(G, "doc")


def get_query_nodes(G: nx.Graph) -> List[str]:
    """Convenience wrapper: return all query node IDs."""
    return get_nodes_by_type(G, "query")
