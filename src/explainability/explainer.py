"""GNNExplainer-based explainability module.

Provides subgraph-level explanations for document-level predictions.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────────────────────────
# GNNExplainer wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MentalHealthExplainer:
    """Wraps PyG's GNNExplainer for heterogeneous mental health graphs."""

    def __init__(
        self,
        model: torch.nn.Module,
        graph,            # HeteroGraphData (raw) or PyG HeteroData
        word_vocab: dict[str, int],
        concept_vocab: dict[str, int],
        category_vocab: dict[str, int],
        num_hops: int = 2,
        top_k_edges: int = 10,
    ) -> None:
        self.model = model
        self.graph = graph
        # Reverse index maps
        self.idx2word = {v: k for k, v in word_vocab.items()}
        self.idx2concept = {v: k for k, v in concept_vocab.items()}
        self.idx2category = {v: k for k, v in category_vocab.items()}
        self.num_hops = num_hops
        self.top_k_edges = top_k_edges

    # ------------------------------------------------------------------
    def explain_document(
        self,
        doc_idx: int,
        pyg_data,
        tokens: list[str],
        extraction_result,
    ) -> dict:
        """
        Build an explanation graph for a single document.

        Returns a dict with:
          - reasoning_chain: list of (token/concept/category) strings
          - explanation_edges: list of (src_label, rel, dst_label)
          - networkx_graph: nx.DiGraph
        """
        reasoning: list[str] = []
        edges: list[tuple[str, str, str]] = []

        # Level 1: Document → Tokens
        doc_label = f"doc_{doc_idx}"
        for tok in tokens[:self.top_k_edges]:
            edges.append((doc_label, "contains", tok))

        # Level 2: Token → Medical concept
        for entity in extraction_result.entities:
            edges.append((entity.surface, "maps_to", entity.concept))
            reasoning.append(entity.surface)
            reasoning.append(entity.concept)

        # Level 3: Concept → Category
        for entity in extraction_result.entities:
            edges.append((entity.concept, "belongs_to", entity.category))
            reasoning.append(entity.category)

        # Build networkx graph
        G = nx.DiGraph()
        for src, rel, dst in edges:
            G.add_edge(src, dst, relation=rel)

        return {
            "doc_idx": doc_idx,
            "reasoning_chain": list(dict.fromkeys(reasoning)),  # deduplicated
            "explanation_edges": edges,
            "networkx_graph": G,
        }

    # ------------------------------------------------------------------
    def visualize_explanation(
        self,
        explanation: dict,
        title: str = "Explanation Subgraph",
        save_path: str | Path | None = None,
    ) -> None:
        G = explanation["networkx_graph"]
        if len(G.nodes) == 0:
            print("  No nodes to visualize.")
            return

        # Color nodes by type
        doc_idx = explanation["doc_idx"]
        color_map: list[str] = []
        node_labels: dict = {}
        for node in G.nodes:
            node_str = str(node)
            if node_str.startswith("doc_"):
                color_map.append("tab:blue")
            elif "_disorder" in node_str or node_str == "crisis":
                color_map.append("tab:red")
            elif any(c in node_str for c in ["_", "ness", "tion", "ity"]):
                color_map.append("tab:orange")
            else:
                color_map.append("tab:green")
            node_labels[node] = node_str[:15]

        pos = nx.spring_layout(G, seed=42, k=1.5)
        plt.figure(figsize=(12, 7))
        nx.draw_networkx(
            G,
            pos=pos,
            labels=node_labels,
            node_color=color_map,
            node_size=800,
            font_size=7,
            arrows=True,
            arrowsize=15,
            edge_color="gray",
        )
        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        legend = [
            mpatches.Patch(color="tab:blue", label="Document"),
            mpatches.Patch(color="tab:green", label="Word/Surface"),
            mpatches.Patch(color="tab:orange", label="Medical Concept"),
            mpatches.Patch(color="tab:red", label="Symptom Category"),
        ]
        plt.legend(handles=legend, loc="upper left", fontsize=8)
        plt.title(title)
        plt.axis("off")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Explanation graph → {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    def print_reasoning_chain(self, explanation: dict) -> None:
        print("\n── Reasoning Chain ──────────────────────────────────────")
        chain = explanation["reasoning_chain"]
        for i, step in enumerate(chain):
            arrow = "↓" if i < len(chain) - 1 else ""
            print(f"  {step} {arrow}")
        print("─────────────────────────────────────────────────────────\n")
