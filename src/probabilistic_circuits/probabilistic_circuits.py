# ============================================================
# Imports
# ============================================================
import random
from abc import ABC, abstractmethod

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


class InputNode(Node):
    def __init__(self, feature_idx, mu=0.0, sigma=1.0, device="cpu"):
        super().__init__(children=[])
        self.feature_idx = feature_idx
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.class_params = {}
        self.device = device
        self.feature_name = self.get_feature_names(feature_idx)

    def get_feature_names(self, idx):
        base = [
            # color
            "R_channel", "G_channel", "B_channel",
            # color transforms
            "Lab_L", "Lab_a", "Lab_b",
            "HSV_H", "HSV_S", "HSV_V",
            # edges
            "Sobel_gray", "Laplacian_gray",
        ]
        # gabor orientations
        for theta in [0, 45, 90, 135]:
            base += [f"Gabor_{theta}_real", f"Gabor_{theta}_imag"]
        # conv filter bank
        convs = [
            "Sobel_x", "Sobel_y", "Laplacian", "Sharpen", "Box3", "Box5",
            "Gaussian3", "Gaussian5", "Gaussian7", "Emboss_NE", "Emboss_SE",
            "Prewitt_x", "Prewitt_y", "Diagonal_45", "Diagonal_135",
            "Horizontal_line", "Vertical_line", "Diagonal45_line",
            "Diagonal135_line", "Gabor_dir_0", "Gabor_dir_45",
            "Gabor_dir_90", "Gabor_dir_135",
        ]
        base += convs
        return base[idx] if idx < len(base) else f"Feature_{idx}"

    def fit(self, X):
        """
        Fit Gaussian parameters (mu, sigma) without labels.
        X: np.ndarray or torch.Tensor of shape (B, C, H, W)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        vals = X[:, self.feature_idx].reshape(-1)  # flatten across batch + pixels
        mu = np.median(vals)
        sigma = np.mean(np.abs(vals - mu)) + 1e-6

        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)


    def evaluate_class(self, X, class_idx):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        vals = X[:, self.feature_idx]
        mu, sigma = self.class_params[class_idx]
        mu, sigma = torch.tensor(mu, device=self.device), torch.tensor(sigma, device=self.device)
        return torch.exp(-0.5 * ((vals - mu) / (sigma + 1e-12))**2) / (sigma * (2 * np.pi) ** 0.5)

    def evaluate(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        vals = X[:, self.feature_idx]
        mu, sigma = self.mu.to(vals.device), self.sigma.to(vals.device)
        return torch.exp(-0.5 * ((vals - mu) / (sigma + 1e-12))**2) / (sigma * (2 * np.pi) ** 0.5)


class SumNode(Node):
    def __init__(self, children, weights=None, device="cpu"):
        super().__init__(children)
        n = len(children)
        if weights is None:
            weights = torch.ones(n, device=device) / n
        elif not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.weights = nn.Parameter(weights / weights.sum(), requires_grad=True)

    def evaluate(self, x):
        child_vals = torch.stack([c.evaluate(x) for c in self.children], dim=0)
        return torch.tensordot(self.weights, child_vals, dims=([0], [0]))


class ProductNode(Node):
    def evaluate(self, x):
        return torch.prod(torch.stack([c.evaluate(x) for c in self.children], dim=0), dim=0)


class ClassifierNode(Node):
    """Stack children outputs into logits across class dimension."""
    def evaluate(self, x):
        # Collect child outputs -> (B, H, W)
        child_outputs = [c.evaluate(x) for c in self.children]  
        return torch.stack(child_outputs, dim=1)  # (B, n_classes, H, W)



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


    def visualize(self, direction="TB", layer_sep=2.0, node_sep=5.0, save_path=None):
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
        print_configs(f"Initialized PCNet with {len(self.get_nodes())} nodes, "
                      f"{sum(p.numel() for p in self.params)} trainable parameters")

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
        B, n_classes, H, W = out.shape
        if H != self.input_size[0] or W != self.input_size[1]:
            out = F.interpolate(out, size=self.input_size, mode='bilinear', align_corners=False)
        return out


    def predict(self, inputs):

        with torch.no_grad():
            logits = self.evaluate(inputs)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            return preds.cpu().numpy()


# ============================================================
# EvoPCNet
# ============================================================
class EvoPCNet(PCNet):
    """Evolutionary PCNet with population of circuits"""

    def __init__(
        self, input_size=(1, 28, 28), patch_size=(9, 9), n_classes=2,
        distribution=None, device="cpu", max_depth=4, max_branching=3,
        feature_extractor="handcrafted", seed=42, init_pop_size=10
    ):
        super().__init__(n_classes, distribution, device,
                         max_depth, max_branching,
                         feature_extractor, seed)
        self.init_pop_size = init_pop_size
        self.population = []

    def initialize_population(self, images, masks):
        self.population = []
        for _ in range(self.init_pop_size):
            depth = random.randint(3, self.max_depth)
            branching = random.randint(2, self.max_branching)
            pc = PCNet(
                n_classes=self.n_classes,
                distribution=self.distribution,
                device=self.device,
                max_depth=depth,
                max_branching=branching,
                feature_extractor=self.feature_extractor,
                seed=random.randint(0, 10000),
            )
            self.population.append(pc)

    def mutation(self):
        # TODO: implement mutation operator
        pass

    def crossover(self):
        # TODO: implement crossover operator
        pass

    def ask(self, images, masks):
        if not self.population:
            self.initialize_population(images, masks)
        else:
            for pc in self.population:
                pc.mutation()
        return self.population

    def tell(self, fitnesses):
        # TODO: use fitness values to evolve population
        pass

    


