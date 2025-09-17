import numpy as np
import random
from abc import ABC, abstractmethod
import copy
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.filters import gabor, sobel, laplace
from skimage.color import rgb2lab, rgb2hsv
from torch import nn

import torch
import torchvision.models as models
import torch.nn.functional as F

# ---------------- Node definitions ----------------
class Node(ABC):
    def __init__(self, children=None):
        self.children = children if children is not None else []

    @abstractmethod
    def evaluate(self, x):
        pass
    
class InputNode(Node):
    def __init__(self, feature_idx, mu=0.0, sigma=1.0, device="cpu"):
        super().__init__(children=[])
        self.feature_idx = feature_idx
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.class_params = {}  # {class_label: (mu, sigma)}
        self.device = device
        self.feature_name = self.get_feature_names(feature_idx)

    def get_feature_names(self, idx):
        feature_names = []

        # --- 1. Raw color channels ---
        feature_names += ["R_channel", "G_channel", "B_channel"]

        # --- 2. Color transforms ---
        feature_names += ["Lab_L", "Lab_a", "Lab_b"]
        feature_names += ["HSV_H", "HSV_S", "HSV_V"]

        # --- 3. Edge / texture ---
        feature_names += ["Sobel_gray", "Laplacian_gray"]

        # --- 4. Gabor filters (4 orientations × 2 outputs: real + imag) ---
        orientations = [0, 45, 90, 135]
        for theta in orientations:
            feature_names.append(f"Gabor_theta_{theta}_real")
            feature_names.append(f"Gabor_theta_{theta}_imag")

        # --- 5. Convolutional filter bank ---
        conv_filters = [
            "Sobel_x", "Sobel_y",
            "Laplacian", "Sharpen",
            "Box3", "Box5",
            "Gaussian3", "Gaussian5", "Gaussian7",
            "Emboss_NE", "Emboss_SE",
            "Prewitt_x", "Prewitt_y",
            "Diagonal_45", "Diagonal_135",
            "Horizontal_line", "Vertical_line",
            "Diagonal45_line", "Diagonal135_line",
            "Gabor_dir_0", "Gabor_dir_45", "Gabor_dir_90", "Gabor_dir_135"
        ]
        feature_names += conv_filters

        return feature_names[idx]

    def fit_supervised(self, X, y):
        """
        X: np.ndarray (N, C, H, W) features
        y: np.ndarray (N, H, W) labels
        """
        C = X.shape[1]
        assert self.feature_idx < C, f"Feature idx {self.feature_idx} out of bounds"

        vals = X[:, self.feature_idx, :, :].reshape(-1)   # flatten all pixels for this channel
        labels = y.reshape(-1)

        for c in np.unique(labels):
            class_vals = vals[labels == c]
            mu = np.median(class_vals)
            sigma = np.mean(np.abs(class_vals - mu)) + 1e-6
            self.class_params[c] = (mu, sigma)

        # default to global parameters if no class chosen
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)

    def evaluate_class(self, X, class_idx):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)

        vals = X[:, self.feature_idx, :, :]   # (B,H,W)

        mu, sigma = self.class_params[class_idx]
        mu = torch.tensor(mu, dtype=torch.float32, device=vals.device)
        sigma = torch.tensor(sigma, dtype=torch.float32, device=vals.device)

        probs = torch.exp(-0.5 * ((vals - mu) / (sigma + 1e-12))**2) \
                / (sigma * (2 * np.pi) ** 0.5)
        return probs

    def evaluate(self, X):
        """
        X: torch.Tensor or np.ndarray of shape (B,C,H,W)
        Returns: torch.Tensor (B,H,W)
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)

        vals = X[:, self.feature_idx, :, :]   # (B,H,W)

        # ensure mu, sigma are tensors on same device
        mu = self.mu.to(vals.device)
        sigma = self.sigma.to(vals.device)

        probs = torch.exp(-0.5 * ((vals - mu) / (sigma + 1e-12))**2) \
                / (sigma * (2 * np.pi) ** 0.5)
        return probs

class SumNode(Node):
    def __init__(self, children, weights=None, device="cpu"):
        super().__init__(children)
        self.n_children = len(children)
        if weights is None:
            init_w = torch.ones(self.n_children, device=device) / self.n_children
        else:
            if isinstance(weights, torch.Tensor):
                init_w = weights.detach().clone().float().to(device)
            else:
                init_w = torch.tensor(weights, dtype=torch.float32, device=device)
            init_w = init_w / init_w.sum()
        # make weights a Parameter so optimizer can update them
        self.weights = nn.Parameter(init_w.clone().float(), requires_grad=True)

    def set_weights(self, weights):
        assert len(weights) == self.n_children, "Weights length must match number of children"
        with torch.no_grad():
            w = torch.tensor(weights, dtype=torch.float32, device=self.weights.device)
            w = w / w.sum()
            self.weights.copy_(w)

    def evaluate(self, x):
        # child.evaluate(x) must return torch.Tensor shaped (B, H, W)
        child_vals = torch.stack([child.evaluate(x) for child in self.children], dim=0)  # (n_children, B, H, W)
        # tensordot weights (n_children,) with child_vals -> (B, H, W)
        weighted_sum = torch.tensordot(self.weights, child_vals, dims=([0], [0]))  # (B, H, W)
        return weighted_sum

class ProductNode(Node):
    def __init__(self, children):
        super().__init__(children)

    def evaluate(self, x):
        child_vals = torch.stack([child.evaluate(x) for child in self.children], dim=0)  # (n_children, B, H, W)
        return torch.prod(child_vals, dim=0)  # (B, H, W)
    
class ClassifierNode(Node):
    """
    Stacks its children outputs as logits across the class dimension.
    Each child should evaluate to (B, H, W) logits/scores for that class.
    Returns: logits tensor (B, n_classes, H, W)
    """
    def __init__(self, children):
        super().__init__(children)
        # no internal weights here — children provide class scores

    def evaluate(self, x):
        # child.evaluate(x) -> (B, H, W)
        child_vals = [child.evaluate(x) for child in self.children]  # list of (B, H, W)
        logits = torch.stack(child_vals, dim=1)  # (B, n_classes, H, W)
        return logits

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
        return self.root.evaluate(input)   # (B, H, W)
    
    def get_nodes(self):
        return self.graph.nodes

    def get_edges(self):
        return [(p, c) for p, children in self.graph.edges.items() for c in children]

    
    
    def visualize(self, direction="TB", layer_sep=2.0, node_sep=5.0):
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
        plt.show()

class PCNet(ProbabilisticCircuit):
    def __init__(self,n_classes=2, distribution=None, device="cpu", max_depth=4, max_branching=3, feature_extractor="handcrafted", seed=42):
        
        super().__init__(n_classes, device)
        self.network = None  # Placeholder for a pc network
        self.max_depth = None
        self.max_branching = None
        self.distribution = distribution
        self.n_classes = n_classes
        self.device = device
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.feature_tensor = None
        self.feature_extractor = feature_extractor
        self.seed = seed
        random.seed(self.seed)

    def set_max_depth(self, depth):
        self.max_depth = depth

    def set_max_branching(self, branching):
        self.max_branching = branching

    def pairwise(self, nodes):
        """Group nodes pairwise for ProductNode layer"""
        random.shuffle(nodes)
        paired = []
        for i in range(0, len(nodes)-1, 2):
            paired.append((nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            paired.append((nodes[-1],))
        return paired
    
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
    
    def branchwise_multi_branch(self, nodes, max_reuse=3):
        """
        Group nodes with limited reuse across branches.
        
        Args:
            nodes: list of nodes in the current layer
            max_reuse: maximum times a node can be reused across groups
        
        Returns:
            branched: list of tuples, each tuple is a group of nodes
        """
        import random
        random.shuffle(nodes)

        # Track how many times each node has been used
        usage_count = {node: 0 for node in nodes}
        branched = []

        while True:
            # Collect candidate nodes still allowed
            candidates = [n for n in nodes if usage_count[n] < max_reuse]
            if not candidates:
                break  # no node left to use

            # Sample group size
            group_size = random.randint(1, min(self.max_branching, len(candidates)))
            group = random.sample(candidates, group_size)

            # Update usage
            for g in group:
                usage_count[g] += 1

            branched.append(tuple(group))

        return branched
    
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
    
    def sum_layer(self, current_nodes):
        """Build a SumNode layer"""
        next_nodes = []
        branches = self.branchwise_multi_branch(current_nodes)
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
    
    def init_network(self, current_nodes):
        """Iteratively build the tree layers instead of recursively."""
        nodes = current_nodes
        for depth in trange(1, self.max_depth + 1, desc="Building PCNet"):
            if depth == self.max_depth:
                # Last layer: create root ClassifierNode
                root = ClassifierNode(nodes)
                self.add_node(root)
                for child in nodes:
                    self.add_edge(root, child)
                return root
            else:
                if depth % 2 == 0:
                    nodes = self.prod_layer(nodes)
                else:
                    nodes = self.sum_layer(nodes)
        return root  # fallback, though loop returns at max_depth
    

    def get_filter_bank(self):
        filters = {}

        # --- 1. Edge detectors (Sobel) ---
        filters["sobel_x"] = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=np.float32)

        filters["sobel_y"] = np.array([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=np.float32)

        # --- 2. Laplacian (second derivative, edges) ---
        filters["laplacian"] = np.array([[-1, -1, -1],
                                        [-1,  8, -1],
                                        [-1, -1, -1]], dtype=np.float32)

        # --- 3. Sharpen (high-pass) ---
        filters["sharpen"] = np.array([[ 0, -1,  0],
                                    [-1,  5, -1],
                                    [ 0, -1,  0]], dtype=np.float32)

        # --- 4. Box blur (mean filter) ---
        filters["box3"] = np.ones((3,3), np.float32) / 9.0
        filters["box5"] = np.ones((5,5), np.float32) / 25.0

        # --- 5. Gaussian blur (multi-scale) ---
        g3 = cv2.getGaussianKernel(3, sigma=0.8)
        g5 = cv2.getGaussianKernel(5, sigma=1.2)
        g7 = cv2.getGaussianKernel(7, sigma=2.0)

        filters["gauss3"] = (g3 @ g3.T).astype(np.float32)
        filters["gauss5"] = (g5 @ g5.T).astype(np.float32)
        filters["gauss7"] = (g7 @ g7.T).astype(np.float32)

        # --- 6. Emboss (directional gradients) ---
        filters["emboss_NE"] = np.array([[-2, -1, 0],
                                        [-1,  1, 1],
                                        [ 0,  1, 2]], dtype=np.float32)

        filters["emboss_SE"] = np.array([[ 0,  1, 2],
                                        [-1,  1, 1],
                                        [-2, -1, 0]], dtype=np.float32)

        # --- 7. Edge orientation (Prewitt) ---
        filters["prewitt_x"] = np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]], dtype=np.float32)

        filters["prewitt_y"] = np.array([[-1, -1, -1],
                                        [ 0,  0,  0],
                                        [ 1,  1,  1]], dtype=np.float32)

        # --- 8. Diagonal edge filters ---
        filters["diag_45"] = np.array([[ 0,  1, 2],
                                    [-1, 0, 1],
                                    [-2,-1, 0]], dtype=np.float32)

        filters["diag_135"] = np.array([[ 2,  1,  0],
                                        [ 1,  0, -1],
                                        [ 0, -1, -2]], dtype=np.float32)
        
        # Horizontal line detector (elongated)
        filters["hline"] = np.array([[-1, -1, -1, -1, -1],
                                    [ 2,  2,  2,  2,  2],
                                    [-1, -1, -1, -1, -1]], dtype=np.float32)

        # Vertical line detector (elongated)
        filters["vline"] = np.array([[-1,  2, -1],
                                    [-1,  2, -1],
                                    [-1,  2, -1],
                                    [-1,  2, -1],
                                    [-1,  2, -1]], dtype=np.float32)

        # Diagonal 45° line detector
        filters["diag45"] = np.array([[ 0,  0, -1,  2, -1],
                                    [ 0, -1,  2, -1,  0],
                                    [-1,  2, -1,  0,  0],
                                    [ 2, -1,  0,  0,  0],
                                    [-1,  0,  0,  0,  0]], dtype=np.float32)

        # Diagonal 135° line detector
        filters["diag135"] = np.array([[-1,  0,  0,  0,  0],
                                    [ 2, -1,  0,  0,  0],
                                    [-1,  2, -1,  0,  0],
                                    [ 0, -1,  2, -1,  0],
                                    [ 0,  0, -1,  2, -1]], dtype=np.float32)

        # Elongated Gaussian-like directional filters
        size = 15
        theta_list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for i, theta in enumerate(theta_list):
            kernel = cv2.getGaborKernel((size, size), sigma=3.0, theta=theta,
                                        lambd=10.0, gamma=0.3, psi=0, ktype=cv2.CV_32F)
            filters[f"gabor_dir_{int(np.degrees(theta))}"] = kernel

        return filters
    
    def compute_conv_features(self, images, filters, device="cpu"):
        """
        Compute convolutional feature maps for a batch of images and a bank of filters.

        :param images: np.ndarray or torch.Tensor, shape (N,H,W,3) or (N,3,H,W)
        :param filters: dict[str,np.ndarray] or list[np.ndarray], each filter is (h,w) or (3,h,w)
        :param device: device to run computations on
        :return: torch.Tensor, shape (N, C_out, H_out, W_out)
        """
        # Convert images to torch tensor
        if isinstance(images, list):
            images = torch.from_numpy(np.stack(images, axis=0)).float()
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(device)

        # Make sure images are channels-first
        if images.ndim == 4 and images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2)  # (N,H,W,3) -> (N,3,H,W)

        # Collect filter list
        if isinstance(filters, dict):
            filt_list = list(filters.values())
        else:
            filt_list = filters

        # --- Pad filters to common size ---
        max_h = max(k.shape[0] for k in filt_list)
        max_w = max(k.shape[1] for k in filt_list)

        def pad_to_size(kernel, target_size=(max_h, max_w)):
            h, w = kernel.shape
            H, W = target_size
            out = np.zeros((H, W), dtype=np.float32)
            y0 = (H - h) // 2
            x0 = (W - w) // 2
            out[y0:y0+h, x0:x0+w] = kernel
            return out

        filt_padded = [pad_to_size(k, (max_h, max_w)) for k in filt_list]

        # Convert to torch conv filters -> shape (out_channels, in_channels, H, W)
        filt_tensor = torch.from_numpy(np.stack(filt_padded, axis=0)).float().unsqueeze(1)
        filt_tensor = filt_tensor.repeat(1, 3, 1, 1).to(device)  # replicate for 3 input channels

        # Apply convolution
        conv_out = F.conv2d(images, filt_tensor, padding="same")

        return conv_out
    
    def get_handcrafted_features(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Compute handcrafted feature bank for a list of images.

        :param images: list of np.ndarray, each (H,W,3) RGB in [0,255] or [0,1]
        :return: feature_tensor: np.ndarray (N,C,H,W)
                N = number of images
                C = number of feature channels
                H,W = spatial resolution
        """
        
        def process_one(image: np.ndarray) -> np.ndarray:
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0  # normalize to [0,1]

            # Support both (H,W,3) and (3,H,W)
            if image.shape[-1] == 3:  # channels-last
                image = np.transpose(image, (2,0,1))  # → (3,H,W)

            C, H, W = image.shape
            assert C == 3, f"Expected RGB image with 3 channels, got shape {image.shape}"

            features = []

            # --- 1. Raw color channels ---
            for c in range(3):
                features.append(image[c, ...])  # shape (H,W)

            # Convert to (H,W,3) for color transforms
            image_hw3 = np.transpose(image, (1, 2, 0))  # (H,W,3)

            # --- 2. Color transforms ---
            lab = rgb2lab(image_hw3)  # (H,W,3)
            hsv = rgb2hsv(image_hw3)  # (H,W,3)
            for c in range(3):
                features.append(lab[..., c])
                features.append(hsv[..., c])

            # --- 3. Edge / texture ---
            gray = cv2.cvtColor((image_hw3*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            features.append(sobel(gray))
            features.append(laplace(gray))

            # --- 4. Gabor filters ---
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                filt_real, filt_imag = gabor(gray, frequency=0.2, theta=theta)
                features.append(filt_real)
                features.append(filt_imag)

            # Stack into (C,H,W)
            return np.stack(features, axis=0).astype(np.float32)

        # Process each image
        if self.feature_extractor != "handcrafted":
            feature_tensors = [process_one(img) for img in images]
        else:
            feature_tensors = [process_one(img) for img in tqdm.tqdm(images, desc="Computing handcrafted features")]

        convolutional_filters = [
            self.get_filter_bank()[key] for key in sorted(self.get_filter_bank().keys())
        ]
        convolutional_features = self.compute_conv_features(images, convolutional_filters, device=self.device)
        feature_tensors = np.concatenate([feature_tensors, convolutional_features], axis=1)

        # Stack into (N,C,H,W)
        return np.stack(feature_tensors, axis=0)
    
    def get_features_tensor_with_patches(self, images: list[np.ndarray], patch_size=(16,16)) -> np.ndarray:
        """
        For each image, compute handcrafted features and additional features from patches.
        Patch-based features are computed with the full handcrafted feature pipeline.
        These patch features are then upsampled/replicated to match original image size
        and concatenated with the original full-image features.
        
        :param images: list of np.ndarray, each (H,W,3) RGB
        :param patch_size: tuple (patch_h, patch_w)
        :return: feature_tensor: np.ndarray (N, C_total, H, W)
        """
        # Base full-image features
        base_features = self.get_handcrafted_features(images)  # (N, C_base, H, W)
        N, C_base, H, W = base_features.shape
        patch_h, patch_w = patch_size

        augmented_features = []

        for idx, img in tqdm.tqdm(enumerate(images), desc="Computing patch-augmented features", total=N):
            # Ensure float32 and (3,H,W)
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
            if img.shape[-1] == 3:
                img = np.transpose(img, (2,0,1))  # (3,H,W)

            patch_feature_maps = []

            # Extract patches
            for i in range(0, H, patch_h):
                for j in range(0, W, patch_w):
                    patch = img[:, i:i+patch_h, j:j+patch_w]
                    if patch.shape[1:] != (patch_h, patch_w):
                        continue  # skip incomplete patches

                    # Compute full handcrafted features for this patch
                    patch_hw3 = np.transpose(patch, (1,2,0))  # (H_patch,W_patch,3)
                    patch_cwh = np.transpose(patch_hw3, (2,0,1))  # (3,H_patch,W_patch)
                    patch_feats = self.get_handcrafted_features([patch_cwh])[0]

                    # Append to patch_feature_maps
                    patch_feature_maps.append((i,j,patch_feats))

            # Create empty feature maps for augmented channels
            C_patch = patch_feature_maps[0][2].shape[0] if patch_feature_maps else 0
            patch_aug = np.zeros((C_patch, H, W), dtype=np.float32)

            # Place patch features back into full image
            for i, j, feats in patch_feature_maps:
                patch_aug[:, i:i+patch_h, j:j+patch_w] = feats

            # Concatenate with base full-image features
            combined_features = np.concatenate([base_features[idx], patch_aug], axis=0)
            augmented_features.append(combined_features)

        return np.stack(augmented_features, axis=0)  # (N, C_total, H, W)

    def get_feature_tensor(self, input_images):
        if self.feature_extractor == "handcrafted":
            return self.get_handcrafted_features(input_images)
        elif self.feature_extractor == "patches":
            return self.get_features_tensor_with_patches(input_images, patch_size=(64,64))
        else:
            raise ValueError(f"Unknown feature extractor {self.feature_extractor}")
    
    def set_input_nodes(self, input_images, labels):
        self.feature_tensor = self.get_feature_tensor(input_images)  # (B,C,H,W)
        print("Feature tensor shape:", self.feature_tensor.shape)
        B, C, H, W = self.feature_tensor.shape

        input_nodes = []
        for i in range(C):
            node = InputNode(i, device=self.device)
            node.fit_supervised(self.feature_tensor, labels)
            input_nodes.append(node)
            self.add_node(node)

        return input_nodes

    def fit_supervised(self, images, masks, epochs=5, lr=0.01):
        # Step 1: Build input nodes with supervision
        input_nodes = self.set_input_nodes(images, masks)

        # Step 2: Build circuit structure
        print("Building circuit...")
        self.root = self.init_network(input_nodes)
        self.set_root(self.root)
        print("Circuit built.")
        # Collect trainable parameters: all SumNode weights and any nn.Parameter in other nodes
        params = []
        for node in tqdm.tqdm(self.graph.nodes, desc="Collecting parameters"):
            # if node has attribute 'weights' and it's a nn.Parameter, include it
            if hasattr(node, 'weights') and isinstance(node.weights, torch.nn.Parameter):
                params.append(node.weights)

        optimizer = torch.optim.Adam(params, lr=lr)

        if self.feature_tensor is None:
            self.feature_tensor = torch.from_numpy(
                self.get_feature_tensor(images)
            ).float().to(self.device)
        target = torch.from_numpy(masks).long().to(self.device)  # (B,1,H,W) maybe

        # ensure target shape is (B,H,W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        pbar = tqdm.tqdm(range(epochs), desc="Training PC", ncols=100)
        losses = []
        for epoch in pbar:
            optimizer.zero_grad()
            logits = self.evaluate(self.feature_tensor)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # Update progress bar postfix with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print("Training complete.")
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.show()
    def evaluate(self, data):
        return self.root.evaluate(data)
    
    def predict_mask(self, image):
        """
        images: numpy list or torch tensor already processed to shape (B,C,H,W)
        returns: numpy array (B,H,W) with predicted class ids
        """

        print("Image type:", type(image))
        if isinstance(image, list) or isinstance(image, np.ndarray):
            # if list of numpy images: compute handcrafted features first
            if isinstance(image, list):
                feature_tensor = torch.from_numpy(self.get_feature_tensor(image)).float().to(self.device)
            else:
                # if user passed (B,C,H,W) numpy directly:
                feature_tensor = torch.from_numpy(image).float().to(self.device)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            feature_tensor = self.get_handcrafted_features([image])
        else:
            raise ValueError("Unsupported input type for predict_mask")

        with torch.no_grad():
            logits = self.evaluate(feature_tensor)            # (B, n_classes, H, W)
            preds = torch.softmax(logits, dim=2)               # (B, H, W)
            return preds.cpu().numpy()

class EvoPCNet(PCNet):
    """
    Evolutionary Probabilistic Circuit Network.
    Maintains a population of PCNets, evolves them over generations.
    Uses random variations in max_depth and max_branching to create diversity,
    At each generation, apply Genetic Programming to recombine and mutate circuits.
    :param init_pop_size: initial population size
    """
    def __init__(self, input_size=(1, 28, 28), patch_size=(9,9), n_classes=2, distribution=None, device="cpu", max_depth=4, max_branching=3, feature_extractor="handcrafted", seed=42, init_pop_size=10):
        super().__init__(input_size, patch_size, n_classes, distribution, device, max_depth, max_branching, feature_extractor, seed)
        self.init_pop_size = init_pop_size
        self.population = []

    def initialize_population(self, images, masks):
        self.population = []
        for _ in range(self.init_pop_size):
            max_depth = random.randint(3, self.max_depth)
            max_branching = random.randint(2, self.max_branching)
            pc = PCNet(n_classes=self.n_classes, distribution=self.distribution, device=self.device,
                       max_depth=max_depth, max_branching=max_branching,
                       feature_extractor=self.feature_extractor, seed=random.randint(0,10000))
            self.population.append(pc)

    def mutation(self):
        pass

    def crossover(self):
        pass

    def ask(self, images, masks):
        if not self.population:
            self.initialize_population(images, masks)
        else:
            for pc in self.population:
                pc.mutation()

        return self.population
    
    def tell(self, fitnesses):
        pass

    


