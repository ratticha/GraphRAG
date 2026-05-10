# GraphRAG: Context Retrieval via Graph-Based Ranking

**Course:** 42913 Social and Information Network Analysis — UTS

A reproducible comparison of graph-based link-prediction and retrieval methods, applied to two datasets:
- **Wikipedia Vote Network** (link prediction / ranking)
- **HotpotQA distractor setting** (GraphRAG-style multi-hop retrieval)

---

## Project Overview

This project frames context retrieval as a **link prediction problem** on a graph:

> If your algorithm predicts a strong link between node A and node B, it means B is highly relevant context for A — exactly the retrieval problem in GraphRAG systems.

### Methods Implemented

| Family | Methods |
|--------|---------|
| Heuristic | Common Neighbours, Jaccard Coefficient, Adamic-Adar, Katz Index |
| Embedding | Node2Vec (cosine similarity on learned embeddings) |
| Probabilistic | PageRank, Personalized PageRank (PPR) |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Area under ROC curve (ranking quality) |
| **Average Precision** | Area under Precision-Recall curve |
| **Precision@10** | Fraction of correct links in top-10 predictions |
| **Precision@20** | Fraction of correct links in top-20 predictions |

---

## Installation

### Option A — pip (recommended)

```bash
# Clone / download the repository
cd GraphRAG

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate graphrag
```

---

## Dataset Setup

Datasets are downloaded automatically when you run the notebooks or the download script.

### Option 1: Automatic download (via notebooks)

Simply run Notebook 02 — the datasets are fetched on first use.

### Option 2: Manual download script

```bash
python scripts/download_data.py
```

### Manual download links

| Dataset | URL |
|---------|-----|
| Wikipedia Vote Network | https://snap.stanford.edu/data/wiki-Vote.html |
| HotpotQA (distractor dev) | https://hotpotqa.github.io/ |

Place downloaded files in `data/raw/`.

---

## How to Run the Notebooks

```bash
jupyter notebook
```

Open notebooks in order:

| Notebook | Contents |
|----------|----------|
| `01_repository_setup.ipynb` | Install libraries, verify imports, create folders |
| `02_dataset_exploration.ipynb` | Explore Wiki-Vote and HotpotQA statistics & graphs |
| `03_wikipedia_vote_experiments.ipynb` | Full link-prediction pipeline on Wiki-Vote |
| `04_hotpotqa_graphrag_experiments.ipynb` | GraphRAG retrieval on HotpotQA |
| `05_universal_graphrag_module.ipynb` | Reusable `retrieve()` function demo |

---

## Running Without Notebooks (CLI)

```bash
# Download all datasets
python scripts/download_data.py

# Run the full Wiki-Vote experiment pipeline
python scripts/run_experiments.py

# Skip Node2Vec if not installed
python scripts/run_experiments.py --no-node2vec
```

---

## src/ Module Reference

```
src/
├── data_loader.py          download_wiki_vote(), load_wiki_vote_graph(),
│                           download_hotpotqa(), load_hotpotqa(), split_edges()
├── graph_builder.py        build_undirected(), build_hotpotqa_graph(),
│                           get_nodes_by_type()
├── heuristic_methods.py    common_neighbours(), jaccard_coefficient(),
│                           adamic_adar(), katz_index(), score_pairs_heuristics()
├── embedding_methods.py    train_node2vec(), score_pairs_node2vec()
├── probabilistic_methods.py run_pagerank(), run_personalized_pagerank(),
│                           score_pairs_pagerank(), score_pairs_personalized_pagerank()
├── evaluation.py           evaluate_all(), precision_at_k(), print_results_table()
├── graphrag_retriever.py   retrieve(), retrieve_batch(), build_retrieval_graph()
└── utils.py                set_random_seeds(), save_dataframe(), save_figure()
```

---

## Universal GraphRAG Retriever — Quick Start

```python
from src.graphrag_retriever import retrieve

sentences = [
    "Einstein developed the theory of relativity.",
    "Relativity changed our understanding of space and time.",
    "Newton formulated the laws of motion and gravity.",
    "Curie discovered polonium and radium through radioactivity research.",
    "Einstein won the Nobel Prize for the photoelectric effect.",
]

results = retrieve(
    sentences=sentences,
    query="What did Einstein contribute to science?",
    top_k=3,
    method="ppr",   # options: ppr, pagerank, common_neighbours, jaccard, adamic_adar
)

for sentence, score in results["top_sentences"]:
    print(f"[{score:.4f}] {sentence}")
```

---

## Expected Outputs

After running all notebooks:

```
results/
├── wiki_vote_results.csv       ROC-AUC, AP, Precision@K per method
└── hotpotqa_results.csv        Precision@K for retrieval methods

figures/
├── wiki_vote_degree_dist.png   Degree distribution (log-log)
├── wiki_vote_subgraph.png      Top-30 node subgraph visualisation
├── wiki_vote_all_metrics.png   Bar chart of all evaluation metrics
├── wiki_vote_roc_curves.png    ROC curves for all methods
├── wiki_vote_pr_curves.png     Precision-Recall curves
├── hotpotqa_stats.png          HotpotQA dataset statistics
├── hotpotqa_graph_structure.png Heterogeneous graph visualisation
├── hotpotqa_retrieval_results.png Retrieval Precision@K bar chart
├── hotpotqa_question_graph.png Single-question graph with GT labels
├── graphrag_retrieval_graph.png Retrieval module graph
└── graphrag_method_comparison_heatmap.png Method comparison heatmap
```

---

## Method Suitability Guide

| Method | Wiki-Vote (Link Pred) | HotpotQA (Retrieval) | Notes |
|--------|----------------------|----------------------|-------|
| Common Neighbours | ✓ | ✓ | Fast; poor on sparse graphs |
| Jaccard | ✓ | ✓ | Better than CN when degrees vary |
| Adamic-Adar | ✓ | ✓ | Good for rare shared neighbours |
| Katz Index | ✓ | — | Expensive on large graphs |
| Node2Vec | ✓ | ✓ | Best with sufficient data |
| PageRank | ✓ | ✓ | Good for hub-finding |
| **Personalized PageRank** | ✓ | **Best** | Query-aware multi-hop traversal |

---

## How This Supports the Report and Presentation

| Deliverable | Source |
|-------------|--------|
| Method descriptions and formulae | `src/heuristic_methods.py`, `src/probabilistic_methods.py` |
| Experimental results table | `results/wiki_vote_results.csv`, `results/hotpotqa_results.csv` |
| Figures | `figures/` directory |
| Interpretations | Notebook 03 Step 7, Notebook 04 Step 8 |
| Code appendix | All files in `src/` |
| Reusable module demo | Notebook 05 |

---

## References

1. Edge, D., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130* (2024).
2. Pan, S., et al. "Unifying Large Language Models and Knowledge Graphs: A Roadmap." *IEEE TKDE* (2024).
3. Leskovec, J., et al. "Predicting Positive and Negative Links in Online Social Networks." *WWW* (2010).
4. Grover, A. & Leskovec, J. "node2vec: Scalable Feature Learning for Networks." *KDD* (2016).
5. Page, L., et al. "The PageRank Citation Ranking: Bringing Order to the Web." *Stanford TR* (1998).
6. Yang, Z., et al. "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." *EMNLP* (2018).
