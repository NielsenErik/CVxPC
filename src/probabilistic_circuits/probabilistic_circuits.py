import numpy as np
import random
from abc import ABC, abstractmethod
import copy
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
import torch
import torch.nn.functional as F
# ---------------- Node definitions ----------------
class Node(ABC):
    def __init__(self, children=None):
        self.children = children if children is not None else []

    @abstractmethod
    def evaluate(self, x):
        pass

class ChannelInputNode(Node):
    def __init__(self, channel_idx, distribution):
        super().__init__(children=[])
        self.channel_idx = channel_idx
        self.distribution = distribution  # torch-based PDF

    def evaluate(self, x):
        # x shape: (B, C, H, W)
        vals = x[:, self.channel_idx, :, :]  # (B, H, W)
        return self.distribution(vals)       # should return (B, H, W)
    
class PatchInputNode(Node):
    def __init__(self, patch_size, distribution):
        """
        patch_size: tuple (ph, pw) specifying patch dimensions
        distribution: callable mapping (B, ph, pw) -> (B,) probabilities
        """
        super().__init__(children=[])
        self.patch_size = patch_size
        self.distribution = distribution
        self.mu, self.std = None, None  # to be set during fitting  

    def extract_patches(self, x, patch_size):
        """
        x: (B, C, H, W)
        patch_size: (ph, pw)
        Returns: patches of shape (B, N, D), where
                N = H*W, D = C*ph*pw
        """
        B, C, H, W = x.shape
        ph, pw = patch_size
        patches = F.unfold(x, kernel_size=(ph, pw), padding=(ph//2, pw//2))  # (B, C*ph*pw, H*W)
        patches = patches.transpose(1, 2)  # (B, H*W, C*ph*pw)
        return patches

    def fit(self, image_tensor, patch_size):
        """
        image_tensor: (B, C, H, W)
        Returns: mu (D,), std (D,) across all patches in the batch
        """
        patches = self.extract_patches(image_tensor, patch_size)  # (B, N, D)
        patches_flat = patches.reshape(-1, patches.shape[-1])  # (B*N, D)
        
        mu = patches_flat.mean(dim=0)
        std = patches_flat.std(dim=0) + 1e-6  # avoid zero std
        return mu, std

    def evaluate(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        patches = self.extract_patches(x, self.patch_size)  # (B, H*W, D)
        
        # Gaussian PDF per patch
        D = patches.shape[-1]
        mu = self.mu.view(1, 1, D)
        std = self.std.view(1, 1, D)
        
        diff = patches - mu
        exponent = -0.5 * (diff / std) ** 2
        probs = torch.exp(exponent).prod(dim=-1) / (std.prod() * (2 * torch.pi) ** (D/2))
        
        return probs.view(B, H, W)
    
class TabularInputNode(Node):
    def __init__(self, feature_idx, distribution):
        super().__init__(children=[])
        self.feature_idx = feature_idx
        self.distribution = distribution  # torch-based PDF
    def evaluate(self, x):
        return self.distribution(x[:, self.feature_idx])
    
class InputNode(Node):
    def __init__(self):
        pass

class SumNode(Node):
    def __init__(self, children, weights=None, device="cpu"):
        super().__init__(children)
        self.n_children = len(children)
        if weights is None:
            weights = torch.ones(self.n_children, device=device) / self.n_children
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.weights /= self.weights.sum()

    def set_weights(self, weights):
        assert len(weights) == self.n_children, "Weights length must match number of children"
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.weights.device)
        self.weights /= self.weights.sum()

    def evaluate(self, x):
        child_vals = torch.stack([child.evaluate(x) for child in self.children], dim=0)  # (n_children, B, H, W)
        weighted_sum = torch.tensordot(self.weights, child_vals, dims=([0], [0]))       # (B, H, W)
        return weighted_sum

class ProductNode(Node):
    def __init__(self, children):
        super().__init__(children)

    def evaluate(self, x):
        child_vals = torch.stack([child.evaluate(x) for child in self.children], dim=0)  # (n_children, B, H, W)
        return torch.prod(child_vals, dim=0)  # (B, H, W)
    
class SoftmaxSumNode(Node):
    def __init__(self, children):
        super().__init__(children)

    def evaluate(self, x):
        child_vals = [child.evaluate(x) for child in self.children]  # list of (B, H, W)
        logits = torch.stack(child_vals, dim=1)  # (B, n_classes, H, W)
        probs = torch.softmax(logits, dim=1)
        return probs

# ---------------- Abstract Graph ----------------
class AbstractGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # parent -> set(children)

    def add_node(self, node):        
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must be added before creating an edge")
        self.edges[parent].add(child)

    def get_children(self, node):
        return self.edges.get(node, set())

    def get_parents(self, node):
        return {p for p, children in self.edges.items() if node in children}

# ---------------- DAG (Acyclic Graph) ----------------
class DirectedAcyclicGraph(AbstractGraph):
    def __init__(self):
        super().__init__()

    def add_edge(self, parent, child):
        super().add_edge(parent, child)
        if self._has_cycle():
            self.edges[parent].remove(child)
            raise ValueError("Adding this edge creates a cycle")

    def _has_cycle(self):
        visited = set()
        stack = set()

        def visit(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for child in self.get_children(node):
                if visit(child):
                    return True
            stack.remove(node)
            return False

        return any(visit(node) for node in self.nodes)

# Probabilistic Circuit
class ProbabilisticCircuit:
    """
    Probabilistic Circuit for image segmentation.
    Evaluates pixel-wise likelihoods for class labels.
    """
    def __init__(self, n_classes=2, device="cpu"):
        self.graph = DirectedAcyclicGraph()
        self.root = None
        self.n_classes = n_classes
        self.device = device

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

    def evaluate(self, input: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the segmentation PC.
        image_tensor: torch.Tensor of shape (B, C, H, W)
        Returns: torch.Tensor of shape (B, n_classes, H, W)
        """
        if self.root is None:
            raise ValueError("Root not set")
        return self.root.evaluate(input)
    
    def predict_mask(self, image_tensor):
        """
        Returns segmentation mask from PC evaluation.
        """
        probs = self.evaluate(image_tensor)  # (B, n_classes, H, W)
        return torch.argmax(probs, dim=2)    # (B, H, W)
    
    def get_nodes(self):
        return self.graph.nodes

    def get_edges(self):
        return [(p, c) for p, children in self.graph.edges.items() for c in children]

    def visualize(self):
        """
        Visualize the circuit using networkx.
        """
        G = nx.DiGraph()
        for node in self.get_nodes():
            label = node.__class__.__name__
            if isinstance(node, InputNode):
                label += f"\n(feature={node.channel_idx})"
            elif isinstance(node, SumNode):
                label += f"\n(weights={np.round(node.weights, 2)})"
            G.add_node(node, label=label)

        for parent, child in self.get_edges():
            G.add_edge(parent, child)

        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color="lightblue")
        plt.show()

class ProbabilisticCircuitNetwork(ProbabilisticCircuit):
    def __init__(self, input_size, patch_size=(9,9), n_classes=2, distribution=None, device="cpu", max_depth=4, max_branching=3):
        
        super().__init__(n_classes, device)
        self.input_size = input_size  # (C, H, W)
        self.patch_size = patch_size  # (ph, pw)
        self.network = None  # Placeholder for a pc network
        self.max_depth = None
        self.max_branching = None
        self.distribution = distribution
        self.n_classes = n_classes
        self.device = device
        self.max_depth = max_depth
        self.max_branching = max_branching

    def pairwise(self, nodes):
        """Group nodes pairwise for ProductNode layer"""
        random.shuffle(nodes)
        paired = []
        for i in range(0, len(nodes)-1, 2):
            paired.append((nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            paired.append((nodes[-1],))
        return paired
    
    def branchwise(self, nodes):
        """Group nodes randomly with max_branching for SumNode layer"""
        random.shuffle(nodes)
        branched = []
        current = nodes[:]
        while len(current) > self.max_branching:
            group_size = random.randint(1, self.max_branching)
            branched.append(tuple(current[:group_size]))
            current = current[group_size:]
        if len(current) > 0:
            branched.append(tuple(current))
        return branched
    
    def prod_layer(self, current_nodes):
        """Build a ProductNode layer"""
        next_nodes = []
        pairs = self.pairwise(current_nodes)
        for pair in pairs:
            if len(pair) == 1:
                next_nodes.append(pair[0])
                continue
            prod_node = ProductNode(children=list(pair))
            self.add_node(prod_node)
            for child in pair:
                self.add_edge(prod_node, child)
            next_nodes.append(prod_node)
        return next_nodes
    
    def sum_layer(self, current_nodes):
        """Build a SumNode layer"""
        next_nodes = []
        branches = self.branchwise(current_nodes)
        for branch in branches:
            # Random weights
            weights = torch.rand(len(branch), device=self.device)

            # Normalize so sum = 1
            weights = weights / weights.sum()
            sum_node = SumNode(children=list(branch), weights=weights, device=self.device)
            self.add_node(sum_node)
            for child in branch:
                self.add_node(child)
                self.add_edge(sum_node, child)
            next_nodes.append(sum_node)
        return next_nodes
    
    def set_next_layer(self, current_nodes, depth):
        """Recursively build the tree layers"""
        if depth == self.max_depth:
            # If multiple nodes remain, create a root SumNode
            if len(current_nodes) == 1:
                root = SoftmaxSumNode(current_nodes)
                self.add_node(root)
                self.add_edge(root, current_nodes[0])
                return root
            else:
                root = SoftmaxSumNode(current_nodes)
                self.add_node(root)
                for child in current_nodes:
                    self.add_edge(root, child)
                return root
        else:
            if depth % 2 == 0:
                next_nodes = self.prod_layer(current_nodes)
            else:
                next_nodes = self.sum_layer(current_nodes)
            return self.set_next_layer(next_nodes, depth+1)
    
    def initialize_network(self, data):
        input_nodes = []
        n_pathces = int(self.input_size[1] * self.input_size[2]/(self.patch_size[0]*self.patch_size[1]))
        for i in range(n_pathces):
            in_node = PatchInputNode(self.patch_size, self.distribution)
            mu, std = in_node.fit(data, self.patch_size)
            in_node.mu = mu
            in_node.std = std
            input_nodes.append(in_node)
            self.add_node(in_node)
        # Simple structure: all inputs to a single sum node, then to a product 
        root = self.set_next_layer(input_nodes, depth=1)
        self.set_root(root)
        

    def fit(self, data, epochs=10, lr=0.001):
    # Initialize network if not already done
        if self.network is None:
            self.initialize_network(data)

        # Initialize weights for SumNodes only if they are uninitialized
        for node in self.network.nodes():
            if isinstance(node, SumNode) and node.get_weights() is None:
                new_weights = torch.rand(len(node.children), device=self.device)
                new_weights = new_weights / new_weights.sum()  # normalize
                node.set_weights(new_weights)

        # EM-like weight update
        for _ in range(epochs):
            # E-step: compute responsibilities given current weights
            probs = self.evaluate(data)  # forward pass to get probabilities
            responsibilities = self.e_step(data, probs)  # soft assignments

            # M-step: update weights
            for node, r in zip(self.network.sum_nodes(), responsibilities):
                # r is the responsibility for this SumNode: shape [num_children]
                new_weights = r / r.sum()  # normalize to sum=1
                # Optional: apply learning rate smoothing
                old_weights = node.get_weights()
                node.set_weights((1 - lr) * old_weights + lr * new_weights)

    def evaluate(self, data):
        return self.root.evaluate(data)
    
    def predict_mask(self, data):
        probs = self.evaluate(data)
        return torch.argmax(probs, dim=2)



if __name__ == "__main__":
    pc_net = ProbabilisticCircuitNetwork(input_size=(1, 28, 28), patch_size=(9,9), n_classes=2, distribution=None, device="cpu", max_depth=4, max_branching=3)

    # Dummy data
    B, C, H, W = 10, 1, 28, 28
    image = torch.rand(B, C, H, W)
    pc_net.initialize_network(image)
    probs = pc_net.evaluate(image)
    mask = pc_net.predict_mask(image)
    probs = probs.detach().cpu().numpy()
    print(mask.shape, probs.shape)
    breakpoint()
    plt.subplot(1, 3, 1)
    plt.imshow(image[0,0].numpy(), cmap='gray')
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask[0].numpy(), cmap='gray')
    plt.title("Predicted Segmentation Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(probs[0,0], cmap='hot')
    plt.title("Predicted Probabilities")
    plt.show()

    pc_net.visualize()