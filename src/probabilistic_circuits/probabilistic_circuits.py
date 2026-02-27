import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ======================================================
# JIT-Compiled Math Kernels (Performance Critical)
# ======================================================
@torch.jit.script
def log_gaussian_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # 0.5 * log(2*pi) ~= 0.9189385
    return -0.5 * ((x - mu) / sigma)**2 - torch.log(sigma) - 0.9189385

@torch.jit.script
def log_laplace_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # b = sigma / sqrt(2). log(2b) = log(sigma*sqrt(2)) = log(sigma) + 0.34657
    b = sigma * 0.70710678
    return -torch.abs(x - mu) / b - (torch.log(sigma) + 0.3465736)

@torch.jit.script
def log_student_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    # Stable Student-T implementation
    z = (x - mu) / (sigma + 1e-12)
    term1 = torch.lgamma((nu + 1) * 0.5)
    term2 = torch.lgamma(nu * 0.5)
    term3 = 0.5 * torch.log(nu * 3.14159265)
    term4 = torch.log(sigma + 1e-12)
    term5 = ((nu + 1) * 0.5) * torch.log1p((z ** 2) / nu)
    return term1 - term2 - term3 - term4 - term5

# ======================================================
# Base Node
# ======================================================
class AbstractNode(nn.Module):
    def __init__(self, child_nodes=None):
        super().__init__()
        # Renamed from 'children' to 'child_nodes' to avoid conflict with nn.Module.children()
        self.child_nodes = nn.ModuleList(child_nodes) if child_nodes else nn.ModuleList()

    def forward(self, x):
        raise NotImplementedError

# ======================================================
# Input Node (No Changes needed, but included for completeness)
# ======================================================
class InputNode(AbstractNode):
    def __init__(self, feature_idx, mu_init=0.0, sigma_init=1.0):
        # InputNodes have no children
        super().__init__()
        self.feature_idx = feature_idx
        
        # Learnable Parameters
        self.mu = nn.Parameter(torch.tensor(float(mu_init)))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))
        self.log_nu = nn.Parameter(torch.log(torch.tensor(5.0))) 
        self.logits = nn.Parameter(torch.zeros(3)) 
        self.gate = nn.Parameter(torch.tensor(2.0)) 

    def fit(self, X):
        """Statistical initialization"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        vals = X[:, self.feature_idx].reshape(-1)
        
        mu = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu)) + 1e-6)
        sigma = mad * 1.4826
        
        with torch.no_grad():
            self.mu.fill_(mu)
            self.log_sigma.fill_(np.log(sigma))
            self.log_nu.fill_(np.log(5.0))
            self.logits.fill_(0.0)

    def forward(self, x):
        vals = x[:, self.feature_idx]
        
        sigma = F.softplus(self.log_sigma) + 1e-6
        nu    = F.softplus(self.log_nu) + 1e-3      
        
        log_gauss = log_gaussian_jit(vals, self.mu, sigma)
        log_lapl  = log_laplace_jit(vals, self.mu, sigma)
        log_stud  = log_student_jit(vals, self.mu, sigma, nu)
        
        # Determine the target shape dynamically
        # If input is (N, C), vals is (N,). Target: (N, 1, 1)
        # If input is (B, C, H, W), vals is (B, H, W). Target: (B, 1, H, W)
        target_shape = [vals.shape[0], 1] + list(vals.shape[1:])
        
        log_gauss = log_gauss.view(*target_shape)
        log_lapl = log_lapl.view(*target_shape)
        log_stud = log_stud.view(*target_shape)

        # Vector of size 3 -> (3, 1, 1...) to match stack
        # Add 1s for Batch, Channel, and Spatial dims
        w_shape = [3] + [1] * len(target_shape)
        w = torch.log_softmax(self.logits, dim=0).view(*w_shape)
        
        # Stack is (3, B, 1, H, W) or (3, N, 1, 1)
        stack = torch.stack([log_gauss, log_lapl, log_stud], dim=0) 
            
        # Broadcasts perfectly over Mixture components (dim=0)
        log_mix = torch.logsumexp(w + stack, dim=0)
        
        return log_mix * torch.sigmoid(self.gate)

# ======================================================
# Sum Node (Updated variable name)
# ======================================================
class SumNode(AbstractNode):
    def __init__(self, child_nodes, weights=None):
        super().__init__(child_nodes)
        n = len(child_nodes)
        if weights is None:
            self.weights = nn.Parameter(torch.full((n,), -np.log(n)))
        else:
            w_t = torch.tensor(weights, dtype=torch.float32)
            self.weights = nn.Parameter(torch.log(w_t / w_t.sum()))

    def forward(self, x):
        # Iterate over self.child_nodes instead of self.children
        child_outputs = [c(x) for c in self.child_nodes]
        
        stack = torch.stack(child_outputs, dim=0)
        
        # log_w initially is (N_children,)
        # We need it to be (N_children, 1, 1...) to broadcast against stack of shape (N_children, Batch, ...)
        w_shape = [-1] + [1] * (stack.dim() - 1)
        log_w = torch.log_softmax(self.weights, dim=0).view(*w_shape)
        
        return torch.logsumexp(stack + log_w, dim=0)

# ======================================================
# Product Node (Updated variable name)
# ======================================================
class ProductNode(AbstractNode):
    def forward(self, x):
        child_outputs = [c(x) for c in self.child_nodes]
        stack = torch.stack(child_outputs, dim=0)
        return torch.sum(stack, dim=0)

# ======================================================
# Classifier / Root Node (Updated variable name)
# ======================================================
class ClassifierNode(AbstractNode):
    def forward(self, x):
        return torch.stack([c(x) for c in self.child_nodes], dim=1)

# ======================================================
# PCNet: The Graph Manager
# ======================================================
class PCNet(nn.Module):
    def __init__(self, n_classes=2, max_depth=4, max_branching=3, seed=42):
        super().__init__()
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.seed = seed
        self.root = None

    def init_network(self, inputs):
        """
        Builds the PC structure based on input data.
        """
        # Reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 1. Create Leaves (InputNodes)
        B, C, H, W = inputs.shape if inputs.dim() == 4 else (inputs.shape[0], inputs.shape[1], 1, 1)
        
        # Create temporary list; they become registered when added to ModuleList/Node
        current_nodes = []
        for i in range(C):
            node = InputNode(i)
            node.fit(inputs) # Initialize stats (Median/MAD)
            current_nodes.append(node)
            
        # Register the leaves for the monolithic structure
        self.leaves = nn.ModuleList(current_nodes)
        
        # 2. Build Random DAG Structure (Bottom-Up)
        nodes = current_nodes
        for depth in range(1, self.max_depth + 1):
            
            # Final Layer: Reduce to N classes
            if depth == self.max_depth:
                nodes = self._reduce_to_n(nodes, self.n_classes)
                self.root = ClassifierNode(nodes)
                break
            
            # Intermediate Layers: Alternate Product & Sum
            if depth % 2 == 0:
                nodes = self._prod_layer(nodes)
            else:
                nodes = self._sum_layer(nodes)

        # Fallback if depth was too small
        if self.root is None:
             self.root = ClassifierNode(self._reduce_to_n(current_nodes, self.n_classes))

    def forward(self, x):
        if self.root is None:
            raise RuntimeError("PCNet not initialized. Call init_network(data) first.")
        return self.root(x)

    # ------------------------------------------------------------------
    # Graph Topology Helpers
    # ------------------------------------------------------------------
    def _pairwise(self, nodes):
        """Shuffle and pair nodes for product layers"""
        pool = list(nodes)
        random.shuffle(pool)
        pairs = []
        while len(pool) >= 2:
            pairs.append((pool.pop(), pool.pop()))
        if pool:
            pairs.append((pool.pop(),))
        return pairs

    def _prod_layer(self, nodes):
        """Create Product Nodes (and pass through some residuals)"""
        next_nodes = []
        for pair in self._pairwise(nodes):
            if len(pair) == 1:
                next_nodes.append(pair[0])
            else:
                next_nodes.append(ProductNode(list(pair)))
        return next_nodes

    def _sum_layer(self, nodes):
        """Create Sum Nodes with random branching factors"""
        next_nodes = []
        pool = list(nodes)
        random.shuffle(pool)
        
        while pool:
            # Random branching factor between 2 and max_branching
            k = random.randint(2, self.max_branching)
            group = []
            for _ in range(k):
                if pool: group.append(pool.pop())
            
            if len(group) == 1:
                next_nodes.append(group[0])
            else:
                s_node = SumNode(group)
                next_nodes.append(s_node)
                
                # Residual Connection: Occasionally pass a child through directly
                # This helps gradient flow in deeper networks
                if random.random() < 0.2:
                    next_nodes.append(group[0])
                    
        return next_nodes

    def _reduce_to_n(self, nodes, n):
        """Force the layer to have exactly n nodes (for the final class output)"""
        if len(nodes) < n:
            # Pad with the last node
            return nodes + [nodes[-1]] * (n - len(nodes))
        
        # Split into n groups and sum them
        chunk_size = int(np.ceil(len(nodes) / n))
        reduced = []
        for i in range(0, len(nodes), chunk_size):
            group = nodes[i : i + chunk_size]
            if len(group) == 1:
                reduced.append(group[0])
            else:
                reduced.append(SumNode(group))
        
        return reduced[:n]

# ======================================================
# Attentional PCNet Nodes
# ======================================================
class AttentionalSumNode(AbstractNode):
    def __init__(self, child_nodes, input_dim, embed_dim=16):
        super().__init__(child_nodes)
        self.n_children = len(child_nodes)
        self.embed_dim = embed_dim
        
        # KEY: Static learned embeddings for each child sub-circuit
        self.keys = nn.Parameter(torch.randn(self.n_children, embed_dim))
        
        # QUERY: Network to project the input into the query space
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.scale = np.sqrt(embed_dim)

    def forward(self, x):
        # 1. VALUES (V): Calculate child log-probabilities normally
        child_outputs = [c(x) for c in self.child_nodes]
        stack = torch.stack(child_outputs, dim=0) # Shape: (N_children, Batch, ...)
        
        # 2. QUERY (Q): Project the raw input
        # Assuming x is (B, C, H, W) or (B, D)
        if x.dim() == 4:
            raw_input = x.mean(dim=(2, 3)) # Global Average Pooling
        else:
            raw_input = x
            
        queries = self.query_proj(raw_input) # Shape: (Batch, embed_dim)
        
        # 3. ATTENTION SCORES: Q dot K^T
        # keys shape: (N_children, embed_dim)
        # scores shape: (Batch, N_children)
        attn_scores = torch.matmul(queries, self.keys.t()) / self.scale
        
        # Transpose to match stack: (N_children, Batch)
        attn_scores = attn_scores.t() 
        
        # Reshape for broadcasting: (N_children, Batch, 1, 1...)
        w_shape = list(attn_scores.shape) + [1] * (stack.dim() - 2)
        log_w = torch.log_softmax(attn_scores, dim=0).view(*w_shape)
        
        # 4. CONTEXT: Dynamic Mixture (LogSumExp)
        return torch.logsumexp(stack + log_w, dim=0)


class AttentionalPCNet(PCNet):
    def __init__(self, in_channels, n_classes=2, max_depth=4, max_branching=3, seed=42, embed_dim=16):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        super().__init__(n_classes=n_classes, max_depth=max_depth, max_branching=max_branching, seed=seed)

    def _sum_layer(self, nodes):
        """Create Attentional Sum Nodes with random branching factors"""
        next_nodes = []
        pool = list(nodes)
        random.shuffle(pool)
        
        while pool:
            # Random branching factor between 2 and max_branching
            k = random.randint(2, self.max_branching)
            group = []
            for _ in range(k):
                if pool: group.append(pool.pop())
            
            if len(group) == 1:
                next_nodes.append(group[0])
            else:
                s_node = AttentionalSumNode(group, input_dim=self.in_channels, embed_dim=self.embed_dim)
                next_nodes.append(s_node)
                
                # Residual Connection
                if random.random() < 0.2:
                    next_nodes.append(group[0])
                    
        return next_nodes

    def _reduce_to_n(self, nodes, n):
        """Force the layer to have exactly n nodes (for the final class output)"""
        if len(nodes) < n:
            # Pad with the last node
            return nodes + [nodes[-1]] * (n - len(nodes))
        
        # Split into n groups and sum them
        chunk_size = int(np.ceil(len(nodes) / n))
        reduced = []
        for i in range(0, len(nodes), chunk_size):
            group = nodes[i : i + chunk_size]
            if len(group) == 1:
                reduced.append(group[0])
            else:
                reduced.append(AttentionalSumNode(group, input_dim=self.in_channels, embed_dim=self.embed_dim))
        
        return reduced[:n]

# ======================================================
# Quick Test Script
# ======================================================
if __name__ == "__main__":
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using Device: {device}")

    # 1. Generate Dummy Data
    B, C, H, W = 8, 3, 64, 64
    data = torch.randn(B, C, H, W).to(device)

    # 2. Initialize Model
    model = PCNet(n_classes=2, max_depth=4)
    
    # 3. Build Structure (Initialize leaves with data stats)
    model.init_network(data)
    
    # 4. Move to GPU
    model = model.to(device)
    
    # 5. Forward Pass (WITH GRADIENTS ENABLED)
    print("Running Forward Pass...")
    # REMOVED: with torch.no_grad():  <-- This was the cause of the error
    output = model(data) 
    
    print(f"✅ Output Shape: {output.shape}")
    print(f"✅ Sample Log-Prob (Pixel 0,0): {output[0, 0, 0, 0].item():.4f}")
    
    # 6. Backward Pass (Verify Differentiability)
    print("Running Backward Pass...")
    
    # We sum the output to get a scalar loss for testing
    loss = output.sum()
    
    # This should now work because the graph was built during forward()
    loss.backward()
    
    # Verify gradients exist on a leaf node
    leaf = None
    for m in model.modules():
        if isinstance(m, InputNode):
            leaf = m
            break
            
    if leaf and leaf.mu.grad is not None:
        print(f"✅ Gradients computed successfully. InputNode.mu.grad norm: {leaf.mu.grad.norm().item():.4f}")
    else:
        print("❌ Gradient check failed.")
# ======================================================
# JIT-Compiled Math Kernels
# ======================================================
@torch.jit.script
def log_gaussian_jit(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes log Gaussian probability.
    x: (B, C, H, W)
    mu: (1, C, K, 1, 1)
    log_sigma: (1, C, K, 1, 1)
    Returns: (B, C, K, H, W)
    """
    # 0.5 * log(2*pi) ~= 0.9189385
    sigma = torch.exp(log_sigma) + 1e-4
    return -0.5 * ((x.unsqueeze(2) - mu) / sigma)**2 - log_sigma - 0.9189385

# ======================================================
# Vectorized Layers
# ======================================================

class GaussianInputLayer(nn.Module):
    """
    Input layer with Mixture of Gaussians per channel.
    Input:  (B, C, H, W)
    Output: (B, C * K, H, W)
    """
    def __init__(self, n_features, n_mixtures=4):
        super().__init__()
        self.n_features = n_features
        self.n_mixtures = n_mixtures
        
        # Parameters: (1, C, K, 1, 1)
        self.mus = nn.Parameter(torch.randn(1, n_features, n_mixtures, 1, 1))
        self.log_sigmas = nn.Parameter(torch.zeros(1, n_features, n_mixtures, 1, 1))
        
    def init_params(self, x):
        """Initialize mus/sigmas from data statistics using Quantiles."""
        # x: (B, C, H, W)
        with torch.no_grad():
            B, C, H, W = x.shape
            # Flatten spatial and batch to get stats per channel
            # We want to spread means across the data range
            for c in range(C):
                vals = x[:, c, :, :].reshape(-1)
                # Compute quantiles to spread means
                q = torch.linspace(0.05, 0.95, self.n_mixtures).to(x.device)
                means = torch.quantile(vals, q)
                
                # Assign to mus
                self.mus.data[0, c, :, 0, 0] = means
                
                # Estimate std based on overall range / mixtures
                # Clamp the std lower bound to prevent torch.log from receiving 0 on static feature maps
                std = max(vals.std().item(), 1e-4) / self.n_mixtures
                self.log_sigmas.data[0, c, :, 0, 0] = torch.log(torch.full((self.n_mixtures,), std))

    def forward(self, x):
        if x.dim() == 2: # Handle flattened case just in case
             x = x.view(x.shape[0], self.n_features, 1, 1)
             
        # log_prob: (B, C, K, H, W)
        log_prob = log_gaussian_jit(x, self.mus, self.log_sigmas)
        
        B, C, K, H, W = log_prob.shape
        # Flatten C, K -> (B, C*K, H, W)
        return log_prob.view(B, C * K, H, W)

class SumLayer(nn.Module):
    """
    Log-Space 1x1 Convolution.
    Mixes C_in channels to C_out channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels) * 0.01)
        
    def forward(self, x):
        # x: (B, C_in, H, W)
        # Weights normalized along dim=1 (sum over inputs = 1)
        log_w = torch.log_softmax(self.weights, dim=1)
        
        # Compute: log( sum_j exp(x_j + w_ij) )
        # Broadcast: (1, Out, In, 1, 1) + (B, 1, In, H, W)
        # Sum over In (dim=2)
        return torch.logsumexp(log_w.view(1, self.out_channels, self.in_channels, 1, 1) + 
                               x.unsqueeze(1), dim=2)

class ProductLayer(nn.Module):
    """
    Spatial Pooling (2x2).
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Sum of log-probs in 2x2 window
        # AvgPool computes Sum/4, so we multiply by 4
        return self.pool(x) * 4.0

class SpatialPCNet(nn.Module):
    def __init__(self, in_channels, n_mixtures=10, depth=3, channels_per_layer=32):
        super().__init__()
        self.depth = depth
        
        # 1. Input Leaf Layer
        self.input_layer = GaussianInputLayer(in_channels, n_mixtures)
        current_c = in_channels * n_mixtures
        
        self.layers = nn.ModuleList()
        
        # 2. Build Pyramid
        for d in range(depth):
            # Sum Layer: Mixes channels
            sum_layer = SumLayer(current_c, channels_per_layer)
            self.layers.append(sum_layer)
            
            # Product Layer: Downsamples spatial
            prod_layer = ProductLayer()
            self.layers.append(prod_layer)
            
            current_c = channels_per_layer
            
        # 3. Root Layer
        # Final mix to 1 channel
        self.root = SumLayer(current_c, 1)

    def init_network(self, x):
        """Data-dependent initialization."""
        self.input_layer.init_params(x)

    def forward(self, x):
        # Save original spatial dims for upsampling
        orig_H, orig_W = x.shape[2], x.shape[3]
        
        out = self.input_layer(x)
        
        for layer in self.layers:
            out = layer(out)
            
        # Final mix
        out = self.root(out) # (B, 1, H', W')
        
        # Instead of summing to scalar, we return the spatial map.
        # If the map is smaller than input, we can upsample it to serve as a pixel-wise score map.
        # This is an approximation for anomaly detection: we distribute the region's score to its pixels.
        
        if out.shape[2] != orig_H or out.shape[3] != orig_W:
            out = F.interpolate(out, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            
        return out # (B, 1, H, W)

class RobustInputLayer(nn.Module):
    """
    Input layer that wraps the robust `InputNode` (Gauss, Laplace, Student-T).
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.leaves = nn.ModuleList([InputNode(i) for i in range(in_channels)])
        
    def init_params(self, x):
        for leaf in self.leaves:
            leaf.fit(x)
            
    def forward(self, x):
        leaf_outputs = [leaf(x) for leaf in self.leaves]
        return torch.cat(leaf_outputs, dim=1) # (B, C, H, W)

class RobustSpatialPCNet(nn.Module):
    """
    A variant of SpatialPCNet that uses the robust `InputNode` 
    (which models Gauss, Laplace, and Student-T components) 
    instead of just Gaussians.
    """
    def __init__(self, in_channels, depth=3, channels_per_layer=32):
        super().__init__()
        self.depth = depth
        
        # 1. Input Leaf Layer
        self.input_layer = RobustInputLayer(in_channels)
        # Note: RobustInputLayer outputs exactly `in_channels`
        # Because the mixtures are integrated within each InputNode
        current_c = in_channels
        
        self.layers = nn.ModuleList()
        
        # 2. Build Pyramid
        for d in range(depth):
            # Sum Layer: Mixes channels
            sum_layer = SumLayer(current_c, channels_per_layer)
            self.layers.append(sum_layer)
            
            # Product Layer: Downsamples spatial
            prod_layer = ProductLayer()
            self.layers.append(prod_layer)
            
            current_c = channels_per_layer
            
        # 3. Root Layer
        # Final mix to 1 channel
        self.root = SumLayer(current_c, 1)

    def init_network(self, x):
        """Data-dependent initialization."""
        self.input_layer.init_params(x)

    def forward(self, x):
        # Save original spatial dims for upsampling
        orig_H, orig_W = x.shape[2], x.shape[3]
        
        out = self.input_layer(x)
        
        for layer in self.layers:
            out = layer(out)
            
        # Final mix
        out = self.root(out) # (B, 1, H', W')
        
        if out.shape[2] != orig_H or out.shape[3] != orig_W:
            out = F.interpolate(out, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            
        return out # (B, 1, H, W)

class HybridPCNet(nn.Module):
    """
    Implements a Monolithic Coarse-to-Fine (Cascade) Anomaly Detection Pipeline.
    Instead of two disjoint networks, this network shares its base `InputNode`s.
    1. Spatial Pyramid: Pools shared leaf nodes spatially to find candidate anomalous regions.
    2. Fine Root: Runs strict, complex pixel-wise operations on flagged anomalous regions.
    """
    def __init__(self, in_channels, spatial_mixtures=10, spatial_depth=3, max_depth=4, channels_per_layer=32):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_depth = spatial_depth
        
        # 1. The fine-grained pixel network (houses the shared Leaves)
        self.pixel_pc = PCNet(n_classes=2, max_depth=max_depth)
        
        # 2. Build the Coarse Spatial Pyramid
        # The base PCNet produces `in_channels` leaf outputs (1 mixture per channel since InputNode handles them)
        current_c = in_channels
        self.spatial_layers = nn.ModuleList()
        
        for d in range(spatial_depth):
            # Sum Layer: Mixes channels
            self.spatial_layers.append(SumLayer(current_c, channels_per_layer))
            # Product Layer: Downsamples spatial (2x2)
            self.spatial_layers.append(ProductLayer())
            current_c = channels_per_layer
            
        # Final spatial mix to 1 channel score
        self.spatial_root = SumLayer(current_c, 1)

    def init_network(self, x):
        """Initializes the base PCNet leaves and structure."""
        self.pixel_pc.init_network(x)
        
    def forward(self, x, refine_quantile=0.10):
        """
        x: (B, C, H, W)
        refine_quantile: Fraction of the most anomalous pixels to refine.
        """
        B, C, H, W = x.shape
        
        # 1. Evaluate Shared Leaves (InputNodes)
        # outputs shape (B, C, H, W) representing log-probs of each input feature
        leaf_outputs = [leaf(x) for leaf in self.pixel_pc.leaves]
        leaf_stack = torch.cat(leaf_outputs, dim=1) # (B, C, H, W)
        
        # 2. Coarse Spatial Pass
        out = leaf_stack
        for layer in self.spatial_layers:
            out = layer(out)
            
        coarse_log_prob = self.spatial_root(out) # (B, 1, H', W')
        
        # Upsample coarse map to original resolution to assign scores pixel-wise
        if coarse_log_prob.shape[2] != H or coarse_log_prob.shape[3] != W:
            coarse_log_prob = F.interpolate(coarse_log_prob, size=(H, W), mode='bilinear', align_corners=False)
        
        # Initialize blended final map
        final_map = coarse_log_prob.clone()
        
        # 3. Identify Anomalous Pixels dynamically via Quantile
        flat_coarse = coarse_log_prob.view(B, -1)
        q_thresh = torch.quantile(flat_coarse, q=refine_quantile, dim=1, keepdim=True).view(B, 1, 1, 1)
        refine_mask = (coarse_log_prob <= q_thresh) # (B, 1, H, W)
        
        if not refine_mask.any():
            return final_map
            
        # 4. Extract and Refine Anomalous Pixels
        for b in range(B):
            mask_b = refine_mask[b, 0] # (H, W)
            if not mask_b.any():
                continue
                
            # Extract raw pixels: (C, H, W) -> (N_pixels, C)
            pixels = x[b].permute(1, 2, 0)[mask_b] 
            
            # Reshape for PCNet InputNode fallback processing: (N, C, 1, 1)
            # Alternatively, we could manually pass leaf_outputs, but passing raw
            # pixels through `self.pixel_pc` is cleaner since `PCNet.forward()`
            # handles the full bottom-up traversal natively on whatever shape we give it.
            pixels_reshaped = pixels.view(-1, C, 1, 1)
            
            # Forward pass through the exact same PyTorch Graph
            fine_out = self.pixel_pc(pixels_reshaped) 
            
            # Marginalize out the class splits -> (N_pixels,)
            fine_log_prob = torch.logsumexp(fine_out, dim=1).view(-1) 
            
            # Replace coarse scores with fine-grained evidence
            final_map[b, 0, mask_b] = fine_log_prob
            
        return final_map

class RandomSpatialPCEnsemble(nn.Module):
    """
    An optimized, fully-differentiable ensemble of SpatialPCNets.
    Initializes models sequentially on the target device, avoiding joblib multiprocessing overhead
    and pickling issues.
    """
    def __init__(self, n_models=10, device="cpu", **model_params):
        """
        Args:
            n_models: Number of random SpatialPCNets in the ensemble.
            device: 'cpu' or 'cuda'.
            **model_params: Arguments passed to SpatialPCNet (in_channels, n_mixtures, depth, channels_per_layer).
        """
        super().__init__()
        self.n_models = n_models
        self.target_device = device
        self.model_params = model_params
        
        # Use ModuleList so PyTorch automatically tracks all sub-models
        self.models = nn.ModuleList()

    def init_network(self, X):
        """
        Builds and initializes the entire ensemble sequentially.
        X: Tensor (B, C, H, W).
        """
        # Ensure data is on the correct device for initialization
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.target_device)
        
        in_channels = self.model_params.get("input_channels", X.shape[1])
        n_mixtures = self.model_params.get("n_mixtures", 10)
        depth = self.model_params.get("max_depth", 3)
        channels_per_layer = self.model_params.get("channels_per_layer", 32)
        
        for i in range(self.n_models):
            # 1. Initialize with random structure per model
            # While the structure of SpatialPCNet is actually fixed, if there are random components,
            # this loop handles creating independent instances.
            model = SpatialPCNet(
                in_channels=in_channels,
                n_mixtures=n_mixtures,
                depth=depth,
                channels_per_layer=channels_per_layer
            ).to(self.target_device)
            
            # 2. Structure Learning & Leaf Initialization
            model.init_network(X)
            
            self.models.append(model)
            
        # Ensure entire ensemble is on the target device
        self.to(self.target_device)

    def forward(self, x):
        """
        Forward pass for the ensemble.
        Returns the averaged log-likelihood per pixel.
        """
        # 1. Get log probabilities from each model in the ensemble
        # Each model outputs shape (B, 1, H, W)
        log_probs_list = [m(x) for m in self.models]
        
        # 2. Stack: Output shape becomes (N_Models, B, 1, H, W)
        stack = torch.stack(log_probs_list, dim=0)
        
        # 3. Ensemble Averaging (Log-Sum-Exp trick)
        # log(Mean(P)) = LogSumExp(log_P) - log(N)
        return torch.logsumexp(stack, dim=0) - np.log(self.n_models)

class RandomHybridPCEnsemble(nn.Module):
    """
    An optimized, fully-differentiable ensemble of HybridPCNets.
    Initializes models sequentially on the target device.
    """
    def __init__(self, n_models=10, device="cpu", **model_params):
        """
        Args:
            n_models: Number of random HybridPCNets in the ensemble.
            device: 'cpu' or 'cuda'.
            **model_params: Arguments passed to HybridPCNet.
        """
        super().__init__()
        self.n_models = n_models
        self.target_device = device
        self.model_params = model_params
        
        self.models = nn.ModuleList()

    def init_network(self, X):
        """
        Builds and initializes the hybrid ensemble sequentially.
        X: Tensor (B, C, H, W).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.target_device)
        
        in_channels = self.model_params.get("input_channels", X.shape[1])
        spatial_mixtures = self.model_params.get("spatial_mixtures", 5)
        spatial_depth = self.model_params.get("spatial_depth", 3)
        max_depth = self.model_params.get("max_depth", 4)
        channels_per_layer = self.model_params.get("channels_per_layer", 32)
        
        for i in range(self.n_models):
            # 1. Initialize Hybrid Model
            model = HybridPCNet(
                in_channels=in_channels,
                spatial_mixtures=spatial_mixtures,
                spatial_depth=spatial_depth,
                max_depth=max_depth,
                channels_per_layer=channels_per_layer
            ).to(self.target_device)
            
            # 2. Structure Learning & Leaf Initialization
            model.init_network(X)
            
            self.models.append(model)
            
        self.to(self.target_device)

    def forward(self, x, refine_quantile=0.10):
        """
        Forward pass for the hybrid ensemble.
        Passes down the refine_quantile threshold to each HybridPCNet.
        """
        # 1. Get log probabilities from each hybrid model
        log_probs_list = [m(x, refine_quantile=refine_quantile) for m in self.models]
        
        # 2. Stack: (N_Models, B, 1, H, W)
        stack = torch.stack(log_probs_list, dim=0)
        
        # 3. Ensemble Averaging via Log-Sum-Exp
        return torch.logsumexp(stack, dim=0) - np.log(self.n_models)

class RandomPCEnsemble(nn.Module):
    """
    An optimized, fully-differentiable ensemble of basic PCNets.
    Initializes models sequentially on the target device.
    """
    def __init__(self, n_models=10, device="cpu", **model_params):
        """
        Args:
            n_models: Number of random PCNets in the ensemble.
            device: 'cpu' or 'cuda'.
            **model_params: Arguments passed to PCNet.
        """
        super().__init__()
        self.n_models = n_models
        self.target_device = device
        self.model_params = model_params
        
        self.models = nn.ModuleList()

    def init_network(self, X):
        """
        Builds and initializes the entire ensemble sequentially.
        X: Tensor (B, C, H, W) or (N, C).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.target_device)
        
        n_classes = self.model_params.get("n_classes", 2)
        max_depth = self.model_params.get("max_depth", 4)
        max_branching = self.model_params.get("max_branching", 3)
        seed = self.model_params.get("seed", 42)
        
        for i in range(self.n_models):
            # Pass a different seed to each model so they build different random structures
            model = PCNet(
                n_classes=n_classes,
                max_depth=max_depth,
                max_branching=max_branching,
                seed=seed + i
            ).to(self.target_device)
            
            # Structure Learning & Leaf Initialization
            model.init_network(X)
            
            self.models.append(model)
            
        self.to(self.target_device)

    def forward(self, x):
        """
        Forward pass for the ensemble.
        Returns the averaged log-likelihood.
        """
        log_probs_list = [m(x) for m in self.models]
        stack = torch.stack(log_probs_list, dim=0)
        # Ensemble Averaging (Log-Sum-Exp trick)
        return torch.logsumexp(stack, dim=0) - np.log(self.n_models)

class RandomEnsembleAttentionalPCNet(nn.Module):
    """
    Random Ensemble wrapper for the AttentionalPCNet architecture.
    """
    def __init__(self, n_models=5, device="cpu", **model_params):
        super().__init__()
        self.n_models = n_models
        self.target_device = device
        self.model_params = model_params
        self.models = nn.ModuleList()
        self.ensemble_mode = model_params.get("ensemble_mode", "independent")
        self.diversity_weight = model_params.get("diversity_weight", 1.0)

    def init_network(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.target_device)
        
        in_channels = self.model_params.get("input_channels", X.shape[1] if X.dim() == 4 else X.shape[-1])
        n_classes = self.model_params.get("n_classes", 2)
        max_depth = self.model_params.get("max_depth", 4)
        max_branching = self.model_params.get("max_branching", 3)
        seed = self.model_params.get("seed", 42)
        embed_dim = self.model_params.get("embed_dim", 32)
        
        for i in range(self.n_models):
            # Apply unique seed per architecture permutation
            model = AttentionalPCNet(
                in_channels=in_channels,
                n_classes=n_classes,
                max_depth=max_depth,
                max_branching=max_branching,
                seed=seed + i,
                embed_dim=embed_dim
            ).to(self.target_device)
            
            model.init_network(X)
            self.models.append(model)
            
        self.to(self.target_device)

    def forward(self, x):
        log_probs_list = [m(x) for m in self.models]
        stack = torch.stack(log_probs_list, dim=0)
        
        if self.training:
            if self.ensemble_mode == "independent":
                # Independent Training: completely detaches gradient coupling
                return stack.mean(dim=0)
            elif self.ensemble_mode == "diversity":
                # Joint Training with Diversity Loss (Entropy Regularization)
                # Compute mixture log-probability
                log_mix = torch.logsumexp(stack, dim=0) - np.log(self.n_models)
                
                # Mixture weights for each expert: P(m|x)
                log_w = stack - torch.logsumexp(stack, dim=0, keepdim=True)
                p_w = torch.exp(log_w)
                
                # Negative Entropy Penalty: sum(p * log p). 
                # Minimizing this maximizes entropy (forces diversity)
                penalty = torch.sum(p_w * log_w, dim=0) 
                
                # Subtracting penalty from log-likelihood means -mean(output) in loss computes NLL + weight * penalty
                return log_mix - self.diversity_weight * penalty
            else:
                # Standard Joint Training (Prone to Winner-Takes-All Mode Collapse)
                return torch.logsumexp(stack, dim=0) - np.log(self.n_models)
        else:
            # Ensemble Averaging (LogSumExp Trick) for Test-time evaluation
            return torch.logsumexp(stack, dim=0) - np.log(self.n_models)

