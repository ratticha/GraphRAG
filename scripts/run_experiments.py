"""
Script to run the full Wikipedia Vote Network experiment pipeline.

Loads the graph, splits edges, scores all methods, evaluates, and saves
results to results/ and figures/ without requiring Jupyter.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --no-node2vec
    python scripts/run_experiments.py --config configs/config.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd

from src import data_loader, graph_builder, heuristic_methods
from src import embedding_methods, probabilistic_methods, evaluation
from src.utils import set_random_seeds, setup_logging, save_dataframe, save_figure, load_config


def plot_results(results_df: pd.DataFrame, figures_dir: str = "figures") -> None:
    """Bar chart of all methods across all metrics."""
    metric_cols = [c for c in results_df.columns if c != "method"]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        vals = results_df[metric].fillna(0)
        bars = ax.barh(results_df["method"], vals, color="steelblue", edgecolor="white")
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.set_xlim(0, max(vals.max() * 1.15, 0.01))
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
            )

    fig.suptitle("Wikipedia Vote Network — Link Prediction Results", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "wiki_vote_results.png", figures_dir=figures_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run Wikipedia Vote link-prediction experiments")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-node2vec", action="store_true",
                        help="Skip Node2Vec (useful if node2vec is not installed)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging()
    set_random_seeds(cfg["project"]["seed"])

    # ── Load and split graph ─────────────────────────────────────────────────
    txt_path = data_loader.download_wiki_vote(raw_dir=cfg["paths"]["raw_data"])
    G_directed = data_loader.load_wiki_vote_graph(txt_path)
    G_train_dir, test_pos, test_neg = data_loader.split_edges(
        G_directed,
        test_ratio=cfg["wiki_vote"]["test_ratio"],
        seed=cfg["project"]["seed"],
    )
    G_train = graph_builder.build_undirected(G_train_dir)

    all_pairs = test_pos + test_neg

    # ── Heuristic methods ────────────────────────────────────────────────────
    logger.info("Running heuristic methods ...")
    heuristic_scores = heuristic_methods.score_pairs_heuristics(
        G_train,
        all_pairs,
        beta=cfg["heuristics"]["katz_beta"],
        katz_depth=cfg["heuristics"]["katz_depth"],
    )

    # ── Node2Vec ─────────────────────────────────────────────────────────────
    n2v_scores = {}
    if not args.no_node2vec:
        logger.info("Training Node2Vec ...")
        try:
            n2v_cfg = cfg["node2vec"]
            n2v_result = embedding_methods.train_node2vec(
                G_train,
                dimensions=n2v_cfg["dimensions"],
                walk_length=n2v_cfg["walk_length"],
                num_walks=n2v_cfg["num_walks"],
                p=n2v_cfg["p"],
                q=n2v_cfg["q"],
                window=n2v_cfg["window"],
                seed=cfg["project"]["seed"],
            )
            n2v_scores["node2vec"] = embedding_methods.score_pairs_node2vec(
                n2v_result["embeddings"], all_pairs,
                method=n2v_cfg["score_method"],
            )
        except ImportError:
            logger.warning("node2vec not installed — skipping.")

    # ── PageRank ─────────────────────────────────────────────────────────────
    logger.info("Running PageRank methods ...")
    pr_cfg = cfg["pagerank"]
    pr_scores = probabilistic_methods.score_pairs_pagerank(
        G_train, all_pairs, alpha=pr_cfg["alpha"]
    )
    ppr_scores = probabilistic_methods.score_pairs_personalized_pagerank(
        G_train, all_pairs, alpha=pr_cfg["alpha"]
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_scores = {**heuristic_scores, **n2v_scores,
                  "pagerank": pr_scores, "personalized_pagerank": ppr_scores}

    logger.info("Evaluating all methods ...")
    results_df = evaluation.evaluate_all(
        all_scores, test_pos, test_neg,
        k_values=cfg["evaluation"]["k_values"],
    )
    evaluation.print_results_table(results_df)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_dataframe(results_df, "wiki_vote_results.csv", args.results_dir)
    plot_results(results_df, args.figures_dir)
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
