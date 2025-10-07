# ============================================================
# Imports
# ============================================================
import random
from abc import ABC, abstractmethod
import os

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.stats import norm
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import gabor, laplace, sobel
from torch import nn
from tqdm import trange
from collections import Counter

from utils import print_configs, print_debugging, print_info


# ============================================================
# Loss Functions
# ============================================================

class DiceCELoss(nn.Module):
    """Combination of CrossEntropy and Dice loss"""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=pos_weight)

    def dice_loss(self, pred, target, smooth=1.0):
        if pred.size(1) == 1:  # binary
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()
        else:  # multi-class
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=pred.size(1))
            target = target.permute(0, 3, 1, 2).float()

        intersection = (pred_prob * target).sum(dim=(2, 3))
        denominator = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (denominator + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + 0.5 * self.dice_loss(pred, target)


# ============================================================
# Node Classes
# ============================================================
class Node(ABC):
    def __init__(self, children=None):
        self.children = children if children else []

    @abstractmethod
    def evaluate(self, x):
        pass


import torch
import torch.nn as nn
import numpy as np
from math import pi


# ======================================================
# Input Node: returns log-probability
# ======================================================
class InputNode(nn.Module, Node):
    """
    Input node that automatically learns the best-fitting distribution
    among {Gaussian, Laplace, Student-t}.
    """
    def __init__(self, feature_idx, mu_init=0.0, sigma_init=1.0, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children=[])

        self.feature_idx = feature_idx
        self.device = device

        # Shared location and scale
        self.mu = nn.Parameter(torch.tensor(float(mu_init), dtype=torch.float32, device=device))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init), dtype=torch.float32, device=device)))

        # Optional shape parameter for Student-t
        self.log_nu = nn.Parameter(torch.log(torch.tensor(5.0, dtype=torch.float32, device=device)))  # degrees of freedom

        # Mixture weights over distributions: [Gaussian, Laplace, Student-t]
        self.logits = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))

        # Mixture gate
        self.gate = nn.Parameter(torch.tensor(1.0, device=device))  # learnable

        # Feature name
        self.feature_name = f"feat_{feature_idx}"

    def _log_gaussian(self, x, mu, sigma):
        return -0.5 * ((x - mu) / sigma)**2 - torch.log(sigma * (2 * np.pi)**0.5)

    def _log_laplace(self, x, mu, sigma):
        b = sigma / np.sqrt(2.0)
        return -torch.abs(x - mu) / b - torch.log(2 * b)

    def _log_student(self, x, mu, sigma, nu):
        z = (x - mu) / (sigma + 1e-12)
        return (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * np.pi)
            - torch.log(sigma + 1e-12)
            - ((nu + 1) / 2) * torch.log1p((z ** 2) / nu)
        )
    
    def fit(self, X):
        """
        Initialize the distribution parameters (mu, sigma, nu, mixture weights)
        based on the dataset statistics for this feature channel.

        Args:
            X (torch.Tensor or np.ndarray): shape (B, C, H, W)
        """
        # Convert to NumPy for easy aggregation
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        # Select this feature channel
        vals = X[:, self.feature_idx].reshape(-1)

        # Robust initialization: median + MAD
        mu_init = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu_init)) + 1e-6)
        sigma_init = mad * 1.4826  # consistent with Gaussian std

        # Use moderate ν to start (Student-t)
        nu_init = 5.0

        # Start with uniform mixture over {Gaussian, Laplace, Student-t}
        logits_init = np.zeros(3, dtype=np.float32)

        # Copy into parameters without breaking the computational graph
        with torch.no_grad():
            self.mu.copy_(torch.tensor(mu_init, dtype=torch.float32, device=self.device))
            self.log_sigma.copy_(torch.log(torch.tensor(sigma_init, dtype=torch.float32, device=self.device)))
            self.log_nu.copy_(torch.log(torch.tensor(nu_init, dtype=torch.float32, device=self.device)))
            self.logits.copy_(torch.tensor(logits_init, dtype=torch.float32, device=self.device))


    def evaluate(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        vals = X[:, self.feature_idx]
        sigma = torch.exp(self.log_sigma)
        nu = torch.exp(self.log_nu)
        weights = torch.softmax(self.logits, dim=0)

        log_gauss = self._log_gaussian(vals, self.mu, sigma)
        log_lapl  = self._log_laplace(vals, self.mu, sigma)
        log_stud  = self._log_student(vals, self.mu, sigma, nu)

        # log-sum-exp over distributions
        logpdfs = torch.stack([log_gauss, log_lapl, log_stud], dim=0)
        log_mix = torch.logsumexp(torch.log(weights).view(-1, *([1]*(logpdfs.ndim-1))) + logpdfs, dim=0)
        return log_mix * self.gate # scale by gate



# ======================================================
# Sum Node: log-sum-exp over children
# ======================================================
class SumNode(Node, nn.Module):
    def __init__(self, children, weights=None, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children)
        n = len(children)
        if weights is None:
            weights = torch.ones(n, device=device) / n
        elif not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        # store log-weights (trainable)
        log_w = torch.log(weights / weights.sum())
        self.weights = nn.Parameter(log_w, requires_grad=True)
        self.device = device

    def evaluate(self, x):
        """Compute log(∑_i w_i * p_i) = logsumexp(log w_i + log p_i)."""
        child_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)
        log_w = torch.log_softmax(self.weights, dim=0).view(-1, *([1]*(child_logs.ndim-1)))
        return torch.logsumexp(log_w + child_logs, dim=0)


# ======================================================
# Product Node: sum of log-probabilities
# ======================================================
class ProductNode(Node, nn.Module):
    def __init__(self, children):
        nn.Module.__init__(self)
        Node.__init__(self, children)

    def evaluate(self, x):
        """Compute log(∏_i p_i) = ∑_i log p_i."""
        child_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)
        return torch.sum(child_logs, dim=0)


# ======================================================
# Classifier Node: stack log-likelihoods as class logits
# ======================================================
class ClassifierNode(Node, nn.Module):
    def __init__(self, children):
        nn.Module.__init__(self)
        Node.__init__(self, children)

    def evaluate(self, x):
        """Return (B, n_classes, H, W) log-probs."""
        child_logs = [c.evaluate(x) for c in self.children]
        return torch.stack(child_logs, dim=1)




# ============================================================
# Graph Classes
# ============================================================
class AbstractGraph:
    def __init__(self):
        self.nodes, self.edges = set(), {}

    def add_node(self, node):
        self.nodes.add(node)
        self.edges.setdefault(node, set())

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must be added before edge")
        self.edges[parent].add(child)


class DirectedAcyclicGraph(AbstractGraph):
    def add_edge(self, parent, child):
        super().add_edge(parent, child)
        if self._has_cycle():
            self.edges[parent].remove(child)
            raise ValueError("Adding this edge creates a cycle")

    def _has_cycle(self):
        visited, stack = set(), set()

        def visit(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            if any(visit(c) for c in self.edges.get(node, [])):
                return True
            stack.remove(node)
            return False

        return any(visit(n) for n in self.nodes)


# ============================================================
# Probabilistic Circuit
# ============================================================
class ProbabilisticCircuit:
    def __init__(self, n_classes=2, device="cpu"):
        self.graph = DirectedAcyclicGraph()
        self.root, self.n_classes, self.device = None, n_classes, device

    def add_node(self, node):
        self.graph.add_node(node)

    def add_edge(self, parent, child):
        self.graph.add_edge(parent, child)
        if child not in parent.children:
            parent.children.append(child)

    def set_root(self, node):
        if node not in self.graph.nodes:
            raise ValueError("Root must be in the graph")
        self.root = node

    def evaluate(self, x):
        return self.root.evaluate(x) if self.root else None

    def get_nodes(self):
        return self.graph.nodes

    def get_edges(self):
        return [(p, c) for p, cs in self.graph.edges.items() for c in cs]

    # visualization function unchanged ...


# ============================================================
# PCNet and EvoPCNet (keep as-is but cleaned)
# ============================================================
# [Your PCNet implementation cleaned goes here ... same structure but formatted]


    def visualize(self, direction="LR", layer_sep=2.0, node_sep=5.0, save_path=None):
        """
        Visualize the probabilistic circuit as a tree-like hierarchical layout with proper spacing.

        :param direction: 'TB' (top-to-bottom) or 'LR' (left-to-right)
        :param layer_sep: spacing between layers (higher = more vertical spacing)
        :param node_sep: spacing between nodes in the same layer (higher = more horizontal spacing)
        """
        G = nx.DiGraph()
        colors = {}

        # Add nodes with labels and colors
        for node in self.get_nodes():
            if isinstance(node, InputNode):
                label = "In"
                label += f"\n{node.feature_name}"
                colors[node] = "lightgreen"
            elif isinstance(node, SumNode):
                label = "Sum"
                weights_str = ','.join([f"{w:.2f}\n" for w in node.weights.detach().cpu().numpy()])
                label += f"\n(w={weights_str})"
                colors[node] = "skyblue"
            elif isinstance(node, ProductNode):
                label = "Product"
                colors[node] = "lightcoral"
            elif isinstance(node, ClassifierNode):
                label = "Classifier"
                colors[node] = "orange"
            else:
                colors[node] = "lightgray"
            G.add_node(node, label=label)

        # Add edges
        for parent, child in self.get_edges():
            G.add_edge(parent, child)

        for n in G.nodes:
            G.nodes[n]['rankdir'] = 'TB' if direction == "TB" else 'LR'
        G.graph['graph'] = {'ranksep': str(layer_sep), 'nodesep': str(node_sep)}

        # Hierarchical layout
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        if direction == "LR":
            pos = {k: (y, -x) for k, (x, y) in pos.items()}
        
        labels = nx.get_node_attributes(G, 'label')
        node_colors = [colors[n] for n in G.nodes]

        plt.figure(figsize=(16, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_size=4000,
            node_color=node_colors,
            font_size=7,
            font_color="black",
            edgecolors="k",
            linewidths=0.7,
            arrows=True,
            arrowsize=12
        )
        plt.title("Probabilistic Circuit Tree Visualization", fontsize=16)
        if save_path is not None:
            save_path = os.path.join(save_path, "pc_tree.png")
            plt.savefig(save_path, bbox_inches='tight')
            print_info(f"PC visualization saved to {save_path}")
# ============================================================
# PCNet
# ============================================================
class PCNet(ProbabilisticCircuit):
    def __init__(
        self, input_size=(256, 256), n_classes=2, distribution=None, device="cpu",
        max_depth=4, max_branching=3,
        seed=42
    ):
        super().__init__(n_classes, device)
        self.input_size = input_size
        self.distribution = distribution
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.seed = seed
        random.seed(self.seed)

    # --------------------------------------------------------
    # Network structure helpers
    # --------------------------------------------------------
    def pairwise(self, nodes):
        """Group nodes pairwise for ProductNode"""
        random.shuffle(nodes)
        pairs = [(nodes[i], nodes[i + 1]) for i in range(0, len(nodes) - 1, 2)]
        if len(nodes) % 2 == 1:
            pairs.append((nodes[-1],))
        return pairs

    def prod_layer(self, current_nodes):
        """Build a ProductNode layer"""
        next_nodes = []
        for pair in self.pairwise(current_nodes):
            if len(pair) == 1:
                next_nodes.append(pair[0])
                continue
            prod = ProductNode(list(pair))
            self.add_node(prod)
            for child in pair:
                self.add_edge(prod, child)
            next_nodes.append(prod)
        return next_nodes

    def branchwise(self, nodes):
        """Randomly group nodes with max_branching for SumNode"""
        random.shuffle(nodes)
        groups, current = [], nodes[:]
        while len(current) > self.max_branching:
            gsize = random.randint(2, self.max_branching)
            groups.append(tuple(current[:gsize]))
            current = current[gsize:]
        if current:
            groups.append(tuple(current))
        return groups

    def sum_layer(self, current_nodes):
        """Build a SumNode layer"""
        next_nodes = []
        for branch in self.branchwise(current_nodes):
            weights = torch.rand(len(branch), device=self.device)
            weights = weights / weights.sum()
            node = SumNode(list(branch), weights=weights, device=self.device)
            self.add_node(node)
            for child in branch:
                self.add_node(child)
                self.add_edge(node, child)
            next_nodes.append(node)
        return next_nodes
    
    def reduce_to_n(self, nodes, n_target):
        """
        Force reduce a list of nodes to exactly n_target nodes by grouping.
        """
        if len(nodes) == n_target:
            return nodes
        if len(nodes) < n_target:
            # Pad by duplicating last node
            while len(nodes) < n_target:
                nodes.append(nodes[-1])
            return nodes

        # Too many: group nodes into n_target groups
        groups = np.array_split(nodes, n_target)
        reduced = []
        for g in groups:
            if len(g) == 1:
                reduced.append(g[0])
            else:
                weights = torch.rand(len(g), device=self.device)
                weights = weights / weights.sum()
                node = SumNode(list(g), weights=weights, device=self.device)
                self.add_node(node)
                for child in g:
                    self.add_edge(node, child)
                reduced.append(node)
        return reduced


    def build_network(self, current_nodes):
        nodes = current_nodes
        for depth in range(1, self.max_depth + 1):
            if depth == self.max_depth:
                # Force reduce/pad to exactly n_classes
                nodes = self.reduce_to_n(nodes, self.n_classes)

                print(f"[DEBUG] Final children = {len(nodes)}, expected = {self.n_classes}")
                root = ClassifierNode(nodes)
                self.add_node(root)
                for child in nodes:
                    self.add_edge(root, child)
                return root

            nodes = self.prod_layer(nodes) if depth % 2 == 0 else self.sum_layer(nodes)
        


    def init_network(self, inputs, labels):
        """Initialize the PCNet structure"""
        input_nodes = self.set_input_nodes(inputs, labels)
        root = self.build_network(input_nodes)
        self.set_root(root)
        self.params = [n.weights for n in self.get_nodes() if isinstance(n, SumNode)]
        for i in input_nodes:
            self.params.extend([i.mu, i.log_sigma, i.log_nu, i.logits, i.gate])
        print_configs(f"Initialized PCNet with {len(self.get_nodes())} nodes, "
                      f"{sum(p.numel() for p in self.params)} trainable parameters")
        print_info(self.info())

    # --------------------------------------------------------
    # Training & Prediction
    # --------------------------------------------------------
    def set_input_nodes(self, inputs, labels):
        print_configs("Inputs tensor shape:", inputs.shape)
        B, C, H, W = inputs.shape

        nodes = []
        for i in range(C):
            node = InputNode(i, device=self.device)
            node.fit(inputs)
            nodes.append(node)
            self.add_node(node)
        return nodes
    
    def evaluate(self, x):
    # x should be (C, H, W), convert to (1, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        out = self.root.evaluate(x)
        return out


    def predict(self, inputs):

        with torch.no_grad():
            logits = self.evaluate(inputs)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            return preds.cpu().numpy()
        
    def __repr__(self):
        """
        Return a detailed string representation of the PCNet,
        including architecture info and hierarchical tree structure.
        """
        if not hasattr(self, "graph") or not self.root:
            return "<Uninitialized PCNet>"

        nodes = list(self.get_nodes())

        # Count node types
        counts = {
            "InputNode": sum(isinstance(n, InputNode) for n in nodes),
            "SumNode": sum(isinstance(n, SumNode) for n in nodes),
            "ProductNode": sum(isinstance(n, ProductNode) for n in nodes),
            "ClassifierNode": sum(isinstance(n, ClassifierNode) for n in nodes),
        }

        # Count parameters
        total_params = 0
        param_summary = {}
        for n in nodes:
            if isinstance(n, InputNode):
                p = sum(p.numel() for p in [n.mu, n.log_sigma, n.log_nu, n.logits, n.gate])
                param_summary["InputNode"] = param_summary.get("InputNode", 0) + p
            elif isinstance(n, SumNode):
                p = n.weights.numel()
                param_summary["SumNode"] = param_summary.get("SumNode", 0) + p
            total_params += p if "p" in locals() else 0

        # ---------------------------------------------------------------------
        # HEADER SECTION
        # ---------------------------------------------------------------------
        header = [
            "─────────────────────────────────────────────",
            "🧠  Probabilistic Circuit Network Summary",
            "─────────────────────────────────────────────",
            f"📦 Classes: {self.n_classes}",
            f"📏 Input size: {self.input_size}",
            f"⚙️  Depth: {self.max_depth}, Branching: {self.max_branching}",
            f"💻 Device: {self.device}",
            "─────────────────────────────────────────────",
            "📊 Node counts:",
        ]
        for k, v in counts.items():
            header.append(f"  • {k:<15}: {v}")
        header.append("─────────────────────────────────────────────")
        header.append(f"🧩 Trainable parameters: {total_params}")
        for k, v in param_summary.items():
            header.append(f"  • {k:<15}: {v}")
        header.append("─────────────────────────────────────────────")
        header.append("📚 Structure:")
        header_text = "\n".join(header)

        # ---------------------------------------------------------------------
        # TREE SECTION
        # ---------------------------------------------------------------------
        lines = []

        def format_node(node, prefix="", is_last=True):
            connector = "└─ " if is_last else "├─ "

            # Node label
            if isinstance(node, ClassifierNode):
                label = f"ClassifierNode(classes={len(node.children)})"
            elif isinstance(node, SumNode):
                label = f"SumNode(children={len(node.children)}, weights={len(node.weights)})"
            elif isinstance(node, ProductNode):
                label = f"ProductNode(children={len(node.children)})"
            elif isinstance(node, InputNode):
                label = f"InputNode({node.feature_name})"
            else:
                label = node.__class__.__name__

            lines.append(f"{prefix}{connector}{label}")

            # Recurse for children
            if hasattr(node, "children") and node.children:
                new_prefix = prefix + ("   " if is_last else "│  ")
                for i, child in enumerate(node.children):
                    format_node(child, new_prefix, i == len(node.children) - 1)

        format_node(self.root)

        tree_text = "\n".join(lines)
        return f"{header_text}\n{tree_text}\n─────────────────────────────────────────────"

    
    def info(self, show_layers=True, show_params=True, show_counts=True):
        """
        Print a structured summary of the PCNet architecture.

        Args:
            show_layers (bool): show hierarchical layer summary
            show_params (bool): show learnable parameters per node type
            show_counts (bool): show node counts per type
        """
        print("─────────────────────────────────────────────")
        print("🧠  Probabilistic Circuit Network Summary")
        print("─────────────────────────────────────────────")
        print(f"📦 Classes: {self.n_classes}")
        print(f"📏 Input size: {self.input_size}")
        print(f"⚙️  Depth: {self.max_depth}, Branching: {self.max_branching}")
        print(f"💻 Device: {self.device}")
        print("─────────────────────────────────────────────")

        nodes = list(self.get_nodes())
        total_params = 0

        if show_counts:
            counts = {
                "InputNode": sum(isinstance(n, InputNode) for n in nodes),
                "SumNode": sum(isinstance(n, SumNode) for n in nodes),
                "ProductNode": sum(isinstance(n, ProductNode) for n in nodes),
                "ClassifierNode": sum(isinstance(n, ClassifierNode) for n in nodes),
            }
            print("📊 Node counts:")
            for k, v in counts.items():
                print(f"  • {k:<15}: {v}")
            print("─────────────────────────────────────────────")

        if show_params:
            param_summary = []
            for n in nodes:
                if isinstance(n, InputNode):
                    p = sum(p.numel() for p in [n.mu, n.log_sigma, n.log_nu, n.logits])
                    param_summary.append(("InputNode", p))
                elif isinstance(n, SumNode):
                    p = n.weights.numel()
                    param_summary.append(("SumNode", p))
            total_params = sum(p for _, p in param_summary)
            print(f"🧩 Trainable parameters: {total_params}")
            by_type = {}
            for t, p in param_summary:
                by_type[t] = by_type.get(t, 0) + p
            for t, p in by_type.items():
                print(f"  • {t:<15}: {p}")
            print("─────────────────────────────────────────────")

        return {
            "n_nodes": len(nodes),
            "n_params": total_params,
            "depth": self.max_depth,
            "branching": self.max_branching,
        }

    def save_dict(self):
        """
        Return a compact summary dictionary of the PCNet model.
        This does NOT include the entire structure, only metadata and key stats.
        """
        info = {}

        # --- Basic info ---
        info["model_type"] = self.__class__.__name__
        info["device"] = str(self.device)
        info["n_classes"] = self.n_classes
        info["input_size"] = tuple(self.input_size)
        info["max_depth"] = self.max_depth
        info["max_branching"] = self.max_branching
        info["seed"] = self.seed

        # --- Structure statistics ---
        nodes = list(self.get_nodes())
        edges = list(self.get_edges())
        info["n_nodes"] = len(nodes)
        info["n_edges"] = len(edges)

        type_counts = Counter([n.__class__.__name__ for n in nodes])
        info["node_types"] = dict(type_counts)

        # --- Parameter summary ---
        all_params = []
        mus, sigmas, nus = [], [], []
        sum_weights = []
        input_feats = []

        for node in nodes:
            if isinstance(node, InputNode):
                input_feats.append(node.feature_name)
                mus.append(node.mu.item())
                sigmas.append(torch.exp(node.log_sigma).item())
                nus.append(torch.exp(node.log_nu).item())
            elif isinstance(node, SumNode):
                w = torch.softmax(node.weights, dim=0).detach().cpu().numpy()
                sum_weights.extend(w.tolist())
            # ClassifierNode / ProductNode have no trainable params

        # --- Compute stats safely ---
        def safe_stats(x):
            if len(x) == 0:
                return {"mean": None, "std": None, "min": None, "max": None}
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }

        info["parameter_summary"] = {
            "mu": safe_stats(mus),
            "sigma": safe_stats(sigmas),
            "nu": safe_stats(nus),
            "sum_weights": safe_stats(sum_weights),
        }

        info["n_parameters"] = sum(p.numel() for p in getattr(self, "params", []))
        info["input_features"] = input_feats

        return info


