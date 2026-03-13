"""Explainability module for document-level predictions on hetero graphs.

Primary mode uses PyG's GNNExplainer for faithful edge-importance attribution.
Fallback mode uses a lightweight heuristic chain when explain API is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    _PYG_EXPLAIN_AVAILABLE = True
except ImportError:
    _PYG_EXPLAIN_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# GNNExplainer wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MentalHealthExplainer:
    """Explain predictions on heterogeneous mental health graphs."""

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

        self.explainer = None
        self._wrapped_model = None
        if _PYG_EXPLAIN_AVAILABLE:
            self._init_pyg_explainer()

    # ------------------------------------------------------------------
    def _init_pyg_explainer(self) -> None:
        """Create a PyG Explainer wrapper around the project model interface."""

        class _ModelForExplain(torch.nn.Module):
            def __init__(self, base_model: torch.nn.Module):
                super().__init__()
                self.base_model = base_model

            def forward(self, x_dict, edge_index_dict):
                out = self.base_model(
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    doc_indices=None,
                    alpha=0.0,
                )
                return out["logits"]

        self._wrapped_model = _ModelForExplain(self.model)
        self.explainer = Explainer(
            model=self._wrapped_model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type=None,
            edge_mask_type="object",
            model_config={
                "mode": "multiclass_classification",
                "task_level": "node",
                "return_type": "raw",
            },
        )

    # ------------------------------------------------------------------
    def _label_node(self, ntype: str, idx: int) -> str:
        if ntype == "document":
            return f"doc_{idx}"
        if ntype == "word":
            return self.idx2word.get(idx, f"word_{idx}")
        if ntype == "medical_concept":
            return self.idx2concept.get(idx, f"concept_{idx}")
        if ntype == "symptom_category":
            return self.idx2category.get(idx, f"category_{idx}")
        return f"{ntype}_{idx}"

    # ------------------------------------------------------------------
    def _heuristic_explain(
        self,
        doc_idx: int,
        tokens: list[str],
        extraction_result,
    ) -> dict:
        """Heuristic fallback explanation used when PyG explain is unavailable."""
        reasoning: list[str] = []
        edges: list[tuple[str, str, str]] = []

        doc_label = f"doc_{doc_idx}"
        for tok in tokens[:self.top_k_edges]:
            edges.append((doc_label, "contains", tok))

        for entity in extraction_result.entities:
            edges.append((entity.surface, "maps_to", entity.concept))
            reasoning.append(entity.surface)
            reasoning.append(entity.concept)

        for entity in extraction_result.entities:
            edges.append((entity.concept, "belongs_to", entity.category))
            reasoning.append(entity.category)

        G = nx.DiGraph()
        for src, rel, dst in edges:
            G.add_edge(src, dst, relation=rel)

        return {
            "doc_idx": doc_idx,
            "mode": "heuristic",
            "reasoning_chain": list(dict.fromkeys(reasoning)),
            "explanation_edges": edges,
            "networkx_graph": G,
        }

    # ------------------------------------------------------------------
    def explain_document(
        self,
        doc_idx: int,
        pyg_data,
        tokens: list[str],
        extraction_result,
    ) -> dict:
        """Build explanation graph for one document node."""
        if self.explainer is None:
            return self._heuristic_explain(doc_idx, tokens, extraction_result)

        self.model.eval()
        if self._wrapped_model is not None:
            self._wrapped_model.eval()

        explanation = self.explainer(
            x_dict=pyg_data.x_dict,
            edge_index_dict=pyg_data.edge_index_dict,
            index=int(doc_idx),
        )

        typed_edges: list[tuple[str, str, str, float]] = []
        for etype in pyg_data.edge_types:
            if etype not in explanation.edge_mask_dict:
                continue

            src_type, rel, dst_type = etype
            edge_index = pyg_data[etype].edge_index
            mask = explanation.edge_mask_dict[etype]

            if mask.numel() == 0:
                continue

            k = min(self.top_k_edges, int(mask.numel()))
            top_idx = torch.topk(mask, k=k).indices.cpu()
            for ei in top_idx.tolist():
                score = float(mask[ei].item())
                src_idx = int(edge_index[0, ei].item())
                dst_idx = int(edge_index[1, ei].item())
                src_label = self._label_node(src_type, src_idx)
                dst_label = self._label_node(dst_type, dst_idx)
                typed_edges.append((src_label, rel, dst_label, score))

        # Keep globally strongest edges
        typed_edges.sort(key=lambda x: x[3], reverse=True)
        typed_edges = typed_edges[: self.top_k_edges]

        edges: list[tuple[str, str, str]] = [(s, r, d) for s, r, d, _ in typed_edges]
        G = nx.DiGraph()
        for src, rel, dst, score in typed_edges:
            G.add_edge(src, dst, relation=rel, score=score)

        reasoning = [f"{s} -[{r}:{score:.3f}]-> {d}" for s, r, d, score in typed_edges]
        return {
            "doc_idx": doc_idx,
            "mode": "pyg_gnnexplainer",
            "reasoning_chain": reasoning,
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
