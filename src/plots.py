import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from probabilistic_circuits.probabilistic_circuits import *
from crackdata_processing import *
import pickle
import pandas as pd

# =========================================================
# pcnet_plots.py
# Utility to generate structural and interpretability plots
# for PCNet / MultiPCNet models
# =========================================================

import os
import torch
import numpy as np
import networkx as nx
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------
# 1. STRUCTURAL VISUALIZATIONS
# ---------------------------------------------------------

import matplotlib
import scienceplots
plt.style.use('science')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def safe_graphviz_layout(G, prog="dot"):
    """
    Try pygraphviz → fallback to pydot → fallback to spring layout.
    With Graphviz attributes set for a tighter hierarchical layout.
    """

    # Set default Graphviz attributes on the NetworkX graph
    # These are passed through to both pygraphviz and pydot.
    G.graph["graph"] = {
        "rankdir": "TB",    # top→bottom hierarchy
        "ranksep": "0.1",  # vertical spacing between layers (smaller = tighter)
        "nodesep": "0.1",  # horizontal spacing (smaller = tighter)
        "margin": "0",      # no padding
        "splines": "ortho", # clean orthogonal edges, keeps hierarchy nice
    }
    G.graph["node"] = {
        "width": "0.5",
        "height": "0.5",
        "shape": "box",
        "fontsize": "15",
    }
    G.graph["edge"] = {
        "minlen": "1",      # keep ranks but avoid unnecessarily long edges
        "maxlen": "1.5",
    }

    # Try first: pygraphviz
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog=prog)
    except Exception:
        pass

    # Try second: pydot
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(G, prog=prog)
    except Exception:
        print("[WARN] Falling back to spring_layout (no graphviz).")
        return nx.spring_layout(G)



def plot_pcnet_topology(model, save_path="pcnet_topology.pdf"):
    """
    Draw the PCNet (or MultiPCNet) topology using networkx.
    Works on systems WITHOUT pygraphviz.
    """

    def add_subgraph(G, pcnet, prefix=""):
        # Softer, readable colors
        color_map = {
            "InputNode": "#a6cee3",
            "SumNode": "#1f78b4",
            "ProductNode": "#b2df8a",
            "GateNode": "#33a02c",
            "ResidualNode": "#fb9a99",
            "ClassifierNode": "#e31a1c",
            "SelectorNode": "#cab2d6",
        }

        # Short symbolic labels
        label_map = {
            "InputNode": r"$x_i$",
            "SumNode": r"$\Sigma$",
            "ProductNode": r"$\Pi$",
            "GateNode": r"$G$",
            "ResidualNode": r"$R$",
            "ClassifierNode": r"$C$",
            "SelectorNode": r"$S$",
        }

        for n in pcnet.get_nodes():
            t = n.__class__.__name__
            G.add_node(prefix + str(id(n)),
                       label=label_map.get(t, t),
                       color=color_map.get(t, "#d3d3d3"))

        for parent, child in pcnet.get_edges():
            G.add_edge(prefix + str(id(child)), prefix + str(id(parent)))
        for u, v in G.edges():
            G.edges[u, v]["attr"] = 0.1   # shorter edges (default ~1.0)

    G = nx.DiGraph()

    if hasattr(model, "experts"):
        add_subgraph(G, model.selector, prefix="selector_")
        for i, e in enumerate(model.experts):
            add_subgraph(G, e.model, prefix=f"exp{i}_")
    else:
        add_subgraph(G, model, prefix="pc_")

    pos = safe_graphviz_layout(G, prog="dot")

    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes}

    plt.figure(figsize=(14, 12))
    nx.draw(
        G,
        pos,
        labels=node_labels,
        node_color=node_colors,
        node_size=3600,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        arrowsize=20,
    )
    plt.title("PCNet / MultiPCNet Topology")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()



def compute_node_activation(pcnet, x):
    """Return dict: node -> spatial activation map."""
    acts = {}
    for n in pcnet.get_nodes():
        try:
            a = n.evaluate(x).detach().cpu().numpy()
            acts[n] = _to_spatial_map(a)
        except:
            pass
    return acts


def compute_flow_scores(pcnet, x):
    """
    Computes:
        - node influence score
        - edge flow strengths
    """
    acts = compute_node_activation(pcnet, x)

    # Overall node influence = mean activation magnitude
    node_score = {n: float(np.mean(np.abs(a))) for n, a in acts.items()}

    # Edge flow = child_score / parent_score
    edge_score = {}
    for parent, child in pcnet.get_edges():
        ps = node_score.get(parent, 1e-6)
        cs = node_score.get(child, 1e-6)
        edge_score[(parent, child)] = float(cs / ps)

    return node_score, edge_score


def plot_flow_influence(model, x, save_path="pc_flow.pdf"):
    """
    Visualizes flow of influence inside PCNet/MultiPCNet.
    Thick edges = strong influence.
    Darker nodes = more importance.
    """

    # For MultiPCNet, visualize each expert separately + selector
    if hasattr(model, "experts"):
        plt.figure(figsize=(14, 6))
        plt.suptitle("MultiPCNet Flow Influence")

        # Plot selector
        ax = plt.subplot(1, len(model.experts) + 1, 1)
        pcnet = model.selector
        node_score, edge_score = compute_flow_scores(pcnet, x)
        draw_flow(pcnet, node_score, edge_score, ax, title="Selector")

        # Plot each expert
        for i, e in enumerate(model.experts):
            ax = plt.subplot(1, len(model.experts) + 1, i + 2)
            pcnet = e.model
            node_score, edge_score = compute_flow_scores(pcnet, x)
            draw_flow(pcnet, node_score, edge_score, ax, title=f"Expert {i}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # PCNet case
    plt.figure(figsize=(12, 10))
    pcnet = model

    node_score, edge_score = compute_flow_scores(pcnet, x)

    draw_flow(pcnet, node_score, edge_score, plt.gca(),
              title="PCNet Flow Influence")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_flow(pcnet, node_score, edge_score, ax, title=""):
    """
    Draws a flow graph with:
        - Node color = influence score
        - Edge color = flow score (colormap)
        - Edge width = scaled flow score
        - Node colorbar and edge colorbar
    
    Works with safe_graphviz_layout().
    """
    G = nx.DiGraph()

    # -----------------------------
    # Add nodes + edges
    # -----------------------------
    for n in pcnet.get_nodes():
        G.add_node(str(id(n)), score=node_score[n])

    for parent, child in pcnet.get_edges():
        G.add_edge(str(id(child)),
                    str(id(parent)),
                   flow=edge_score[(parent, child)])

    # -----------------------------
    # Layout
    # -----------------------------
    pos = safe_graphviz_layout(G, prog="dot")

    # -----------------------------
    # Node color normalization
    # -----------------------------
    node_vals = np.array([G.nodes[n]["score"] for n in G.nodes])
    node_min, node_max = node_vals.min(), node_vals.max()
    node_norm = (node_vals - node_min) / (node_max - node_min + 1e-8)

    # -----------------------------
    # Edge color + width normalization
    # -----------------------------
    edges = list(G.edges())
    edge_vals = np.array([G.edges[e]["flow"] for e in edges])
    e_min, e_max = edge_vals.min(), edge_vals.max()
    edge_norm = (edge_vals - e_min) / (e_max - e_min + 1e-8)

    edge_widths = 0.5 + 4.0 * edge_norm   # Width: 0.5 → 4.5
    edge_colors = plt.cm.plasma(edge_norm)  # Colormap for edges
    for u, v in G.edges():
        G.edges[u, v]["len"] = 0.3   # shorter edges (default ~1.0)
    # -----------------------------
    # Draw graph
    # -----------------------------
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_norm,
        cmap="viridis",
        node_size=2400,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=18,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: "" for n in G.nodes},
        font_size=8,
        ax=ax
    )

    ax.set_title(title)
    ax.axis("off")

    # -----------------------------
    # Colorbars (nodes + edges)
    # -----------------------------
    sm_nodes = plt.cm.ScalarMappable(cmap="viridis",
                                     norm=plt.Normalize(node_min, node_max))
    sm_nodes.set_array([])
    cbar1 = plt.colorbar(sm_nodes, ax=ax, fraction=0.04, pad=0.02)
    cbar1.set_label("Node importance", fontsize=8)

    sm_edges = plt.cm.ScalarMappable(cmap="plasma",
                                     norm=plt.Normalize(e_min, e_max))
    sm_edges.set_array([])
    cbar2 = plt.colorbar(sm_edges, ax=ax, fraction=0.04, pad=0.10)
    cbar2.set_label("Edge flow", fontsize=8)



def plot_node_type_distribution(pcnet, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    nodes = list(pcnet.get_nodes())
    types = [n.__class__.__name__ for n in nodes]

    vals = {}
    for t in types:
        vals[t] = vals.get(t,0)+1

    plt.figure(figsize=(6,4))
    plt.bar(vals.keys(), vals.values())
    plt.title("Node Type Distribution")
    plt.ylabel("Number of Nodes")
    plt.savefig(f"{save_dir}/node_type_distribution.pdf", bbox_inches='tight')
    plt.close()


def plot_depth_histogram(pcnet, save_dir):
    """
    Compute the depth of each node by BFS from root.
    """
    os.makedirs(save_dir, exist_ok=True)
    root = pcnet.root
    edges = pcnet.get_edges()

    children_map = {}
    parents_map  = {}
    for p,c in edges:
        children_map.setdefault(p, []).append(c)
        parents_map.setdefault(c, []).append(p)

    # BFS depth
    depths = {root: 0}
    queue = [root]

    while queue:
        n = queue.pop(0)
        for ch in children_map.get(n,[]):
            if ch not in depths:
                depths[ch] = depths[n] + 1
                queue.append(ch)

    plt.figure(figsize=(6,4))
    plt.hist(list(depths.values()))
    plt.title("Node Depth Histogram")
    plt.xlabel("Depth")
    plt.ylabel("Number of Nodes")
    plt.savefig(f"{save_dir}/node_depth_histogram.pdf", bbox_inches='tight')
    plt.close()


def plot_inputnode_parameter_distributions(pcnet, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    mus, sigmas, nus = [], [], []
    for n in pcnet.get_nodes():
        if n.__class__.__name__ == "InputNode":
            mus.append(n.mu.item())
            sigmas.append(torch.exp(n.log_sigma).item())
            nus.append(torch.exp(n.log_nu).item())

    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].hist(mus, bins=20); axs[0].set_title("mu distribution")
    axs[1].hist(sigmas, bins=20); axs[1].set_title("sigma distribution")
    axs[2].hist(nus, bins=20); axs[2].set_title("nu distribution")
    fig.tight_layout()
    plt.savefig(f"{save_dir}/inputnode_param_distributions.pdf")
    plt.close()

# ---------------------------------------------------------
# 2. ACTIVATION & INTERPRETABILITY MAPS
# ---------------------------------------------------------

def eval_all_nodes(pcnet, x):
    """
    Return a dict: node -> activation (log-prob map)
    """
    node_outputs = {}
    # evaluate in topological order
    for n in pcnet.get_nodes():
        try:
            out = n.evaluate(x)
            node_outputs[n] = out.detach().cpu().numpy()
        except:
            pass
    return node_outputs

def plot_class_loglik_maps(pcnet, x, save_dir):
    """
    Plot log-likelihood maps with colorbars, normalization, and
    consistent scales across classes.
    """
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        logits = pcnet.evaluate(x)

    ll = logits[0].detach().cpu().numpy()  # (C,H,W)
    C = ll.shape[0]

    # Normalize for consistent visual comparison
    vmin = np.percentile(ll, 5)
    vmax = np.percentile(ll, 95)

    fig, axs = plt.subplots(1, C, figsize=(6*C, 5))
    if C == 1:
        axs = [axs]

    for c in range(C):
        im = axs[c].imshow(
            ll[c],
            cmap="magma",
            vmin=vmin,
            vmax=vmax
        )
        axs[c].set_title(f"Class {c} log-likelihood")
        axs[c].axis("off")
        fig.colorbar(im, ax=axs[c], fraction=0.045, pad=0.03)
    plt.savefig(f"{save_dir}/class_loglik_maps.pdf", bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# 3. GATE & RESIDUAL INTERPRETABILITY
# ---------------------------------------------------------

def plot_gate_residual_maps(pcnet, x, save_dir):
    """
    Visualize gate α and residual β values as global interpretability signals.
    """
    os.makedirs(save_dir, exist_ok=True)

    gate_vals = []
    resid_vals = []

    for n in pcnet.get_nodes():
        if n.__class__.__name__ == "GateNode":
            gate_vals.append(torch.sigmoid(n.alpha).item())
        if n.__class__.__name__ == "ResidualNode":
            resid_vals.append(torch.sigmoid(n.beta).item())

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].hist(gate_vals); ax[0].set_title("Gate alpha values (sigmoid(alpha))")
    ax[1].hist(resid_vals); ax[1].set_title("Residual beta values")
    plt.savefig(f"{save_dir}/gate_residual_distributions.pdf", bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# 4. GRADIENT-BASED SALIENCY ("CIRCUIT SALIENCY")
# ---------------------------------------------------------

def plot_input_saliencies(pcnet, x, save_dir):
    """
    Compute ∂ log p(x) / ∂ x and always reshape to (H, W) safely.
    Works even if gradient is flattened or has unexpected dims.
    """
    os.makedirs(save_dir, exist_ok=True)

    x = x.clone().detach().requires_grad_(True)

    out = pcnet.evaluate(x)
    loss = out.sum()
    loss.backward()

    grad = x.grad.detach().cpu().numpy()

    # --- Normalize shape: always obtain (C, H, W) ---
    if grad.ndim == 4:
        # (1,C,H,W)
        grad = grad[0]
    elif grad.ndim == 3:
        # (C,H,W)
        pass
    elif grad.ndim == 2:
        # (C, H*W) → reshape
        C, HW = grad.shape
        H = W = int(np.sqrt(HW))
        grad = grad.reshape(C, H, W)
    elif grad.ndim == 1:
        # (H*W,) → assume C=1
        HW = grad.shape[0]
        H = W = int(np.sqrt(HW))
        grad = grad.reshape(1, H, W)
    else:
        print(f"[WARN] Unexpected gradient shape: {grad.shape}")
        return

    C, H, W = grad.shape

    fig, axs = plt.subplots(1, C, figsize=(5*C, 4))

    if C == 1:
        axs = [axs]

    for c in range(C):
        im = axs[c].imshow(np.abs(grad[c]), cmap="inferno")
        axs[c].set_title(f"Saliency channel {c}")
        axs[c].axis("off")
        fig.colorbar(im, ax=axs[c], fraction=0.046, pad=0.04)
    plt.savefig(f"{save_dir}/input_saliencies.pdf", bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# 5. MULTIPCNET INTERPRETABILITY: EXPERT RESPONSIBILITY
# ---------------------------------------------------------

def plot_expert_selection_map(multinet, x, save_dir):
    """
    For soft selectors: extract mixture weights → soft responsibility map.
    """
    os.makedirs(save_dir, exist_ok=True)

    out = multinet.evaluate(x)     # (B,C,H,W)

    if hasattr(multinet.selector, "logits"):
        w = torch.softmax(multinet.selector.logits, dim=0).cpu().numpy()
        plt.figure(figsize=(6,4))
        plt.bar(np.arange(len(w)), w)
        plt.title("Soft selector expert weights")
        plt.savefig(f"{save_dir}/selector_weights.pdf")
        plt.close()
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================================================
# UTIL: convert any activation to a valid (H, W) map
# =========================================================
def _to_spatial_map(arr):
    arr = np.array(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # assume (C, H, W)
        return arr.mean(axis=0)
    if arr.ndim == 1:
        HW = arr.shape[0]
        H = W = int(np.sqrt(HW))
        return arr.reshape(H, W)

    # fallback
    H, W = arr.shape[-2], arr.shape[-1]
    return arr.reshape(-1, H, W).mean(axis=0)


# =========================================================
# PLOTTER: generic 4-panel node visualization
# =========================================================
def plot_generic_node_panel(node, x_batch, save_path):
    """
    New version: supports multiple input images (N,C,H,W)
    Creates N rows, each row = (original, heatmap, overlay)
    """
    x_batch = x_batch.detach().cpu()
    N = x_batch.shape[0]

    # Compute activations for all images
    acts = []
    for i in range(N):
        a = node.evaluate(x_batch[i:i+1]).detach().cpu().numpy()
        acts.append(_to_spatial_map(a))

    fig, axs = plt.subplots(N, 3, figsize=(12, 4*N))
    if N == 1:
        axs = axs.reshape(1, 3)

    for i in range(N):
        img = x_batch[i].numpy().transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img[:, :, 0]

        act = acts[i]
        lo, hi = np.percentile(act, [5, 95])
        act_norm = np.clip((act - lo) / (hi - lo + 1e-8), 0, 1)

        overlay = (0.6 * img + 0.4 * act_norm[..., None]).clip(0, 1)

        # Original
        axs[i, 0].imshow(img, cmap="gray")
        axs[i, 0].set_title(f"Original {i}")
        axs[i, 0].axis("off")

        # Heatmap
        im = axs[i, 1].imshow(act, cmap="viridis")
        axs[i, 1].set_title(f"{node.__class__.__name__} activation {i}")
        axs[i, 1].axis("off")
        fig.colorbar(im, ax=axs[i, 1], fraction=0.05, pad=0.04)

        # Overlay
        axs[i, 2].imshow(overlay)
        axs[i, 2].set_title(f"Overlay {i}")
        axs[i, 2].axis("off")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# =========================================================
# PLOTTER: GateNode specialist (shows both children)
# =========================================================
def plot_gate_node(node, x_batch, save_path):
    N = x_batch.shape[0]
    x_batch = x_batch.detach().cpu()

    left_maps = []
    right_maps = []

    for i in range(N):
        acts = [ _to_spatial_map(child.evaluate(x_batch[i:i+1]).detach().cpu().numpy())
                 for child in node.children ]
        left_maps.append(acts[0])
        right_maps.append(acts[1])

    fig, axs = plt.subplots(N, 4, figsize=(16, 4*N))
    if N == 1:
        axs = axs.reshape(1, 4)

    gate_val = float(torch.sigmoid(node.alpha))

    for i in range(N):
        L = left_maps[i]
        R = right_maps[i]
        lo = min(np.percentile(L,5), np.percentile(R,5))
        hi = max(np.percentile(L,95), np.percentile(R,95))

        Ln = np.clip((L-lo)/(hi-lo+1e-8),0,1)
        Rn = np.clip((R-lo)/(hi-lo+1e-8),0,1)
        diff = Ln - Rn

        axs[i,0].imshow(L, cmap="viridis")
        axs[i,0].set_title(f"Left {i}")
        axs[i,0].axis("off")

        axs[i,1].imshow(R, cmap="viridis")
        axs[i,1].set_title(f"Right {i}")
        axs[i,1].axis("off")

        im = axs[i,2].imshow(diff, cmap="coolwarm")
        axs[i,2].set_title(f"Diff {i}")
        axs[i,2].axis("off")
        fig.colorbar(im, ax=axs[i,2], fraction=0.05, pad=0.04)

        axs[i,3].text(
            0,1,
            f"GateNode\nalpha={gate_val:.3f}\nLeft>Right=red\nRight>Left=blue",
            va="top", fontsize=10
        )
        axs[i,3].axis("off")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



# =========================================================
# PLOTTER: SumNode specialist
# =========================================================
def plot_sum_node(node, x_batch, save_path):
    N = x_batch.shape[0]
    K = len(node.children)

    maps = [ [] for _ in range(K) ]

    for i in range(N):
        for k, child in enumerate(node.children):
            a = child.evaluate(x_batch[i:i+1]).detach().cpu().numpy()
            maps[k].append(_to_spatial_map(a))

    w = torch.softmax(node.weights, dim=0).detach().cpu().numpy()

    fig, axs = plt.subplots(N, 4, figsize=(16, 4*N))
    if N == 1:
        axs = axs.reshape(1,4)

    for i in range(N):
        axs[i,0].imshow(maps[0][i], cmap="viridis")
        axs[i,0].set_title(f"Child 0 (w={w[0]:.2f})")
        axs[i,0].axis("off")

        axs[i,1].imshow(maps[1][i], cmap="viridis")
        axs[i,1].set_title(f"Child 1 (w={w[1]:.2f})")
        axs[i,1].axis("off")

        mix = w[0]*maps[0][i] + w[1]*maps[1][i]
        im = axs[i,2].imshow(mix, cmap="magma")
        axs[i,2].set_title("Weighted output")
        axs[i,2].axis("off")
        fig.colorbar(im, ax=axs[i,2], fraction=0.05, pad=0.04)

        axs[i,3].text(
            0,1,
            f"SumNode\nChildren={K}\nWeights={w}",
            fontsize=10, va="top"
        )
        axs[i,3].axis("off")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_distribution_flow_paths(model, x_batch, save_path="distribution_paths.pdf"):
    """
    NEW PLOT:
    Shows the distribution evolution for each input image (row).
    Columns follow the PCNet computational path:
    Input → InputNodes → internal nodes → ClassifierNode.

    - one row per input image
    - one column per node along the evaluation path
    - activation maps are normalized globally
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_batch = x_batch.detach().cpu()
    B = x_batch.shape[0]

    # ---------------------------------------------------------
    # 1. Get model nodes in topological evaluation order
    # ---------------------------------------------------------
    nodes = list(model.get_nodes())

    # reorder: ensure InputNodes first, ClassifierNodes last
    def order_key(n):
        name = n.__class__.__name__
        if name == "InputNode": return 0
        if name == "ClassifierNode": return 2
        return 1

    nodes = sorted(nodes, key=order_key)

    # ---------------------------------------------------------
    # 2. Evaluate all nodes for each input image
    # ---------------------------------------------------------
    all_maps = []   # shape: B × T × (H,W)
    for img_idx in range(B):
        x = x_batch[img_idx:img_idx+1]

        row_maps = []
        for node in nodes:
            try:
                a = node.evaluate(x).detach().cpu().numpy()
                a = _to_spatial_map(a)
                row_maps.append(a)
            except:
                # broken node → append blank
                row_maps.append(None)

        all_maps.append(row_maps)

    # ---------------------------------------------------------
    # 3. Compute global normalization range (5–95 percentile)
    # ---------------------------------------------------------
    vals = []
    for row in all_maps:
        for m in row:
            if m is not None:
                vals.append(m.flatten())

    if len(vals) == 0:
        print("[WARN] No activations available.")
        return

    vals = np.concatenate(vals)
    lo, hi = np.percentile(vals, [5, 95])

    # ---------------------------------------------------------
    # 4. Plot storyboard:
    #     - 1 row = 1 input image
    #     - columns = Input → InputNodes → ... → ClassifierNode
    # ---------------------------------------------------------
    T = len(nodes)
    fig, axs = plt.subplots(B, T+1, figsize=(3*(T+1), 3*B))

    if B == 1:
        axs = axs.reshape(1, T+1)

    # Column 0 is original image
    for i in range(B):
        img = x_batch[i].numpy().transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img[:, :, 0]

        axs[i,0].imshow(img, cmap="gray")
        axs[i,0].set_title("Input")
        axs[i,0].axis("off")

    # Node activations
    for row_i in range(B):
        for col_j, node in enumerate(nodes):
            ax = axs[row_i, col_j+1]
            act = all_maps[row_i][col_j]

            title = f"{col_j}: {node.__class__.__name__}"

            if act is None:
                ax.set_title(title + "\n(no eval)")
                ax.axis("off")
                continue

            im = ax.imshow(act, cmap="viridis", vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    # Add a single global colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[OK] Saved distribution path storyboard → {save_path}")


# =========================================================
# MASTER DISPATCHER
# =========================================================
def visualize_node(node, x_batch, folder, idx):
    node_type = node.__class__.__name__
    save_path = os.path.join(folder, f"node_{idx}_{node_type}.pdf")

    if node_type == "GateNode":
        plot_gate_node(node, x_batch, save_path)
    elif node_type == "SumNode":
        plot_sum_node(node, x_batch, save_path)
    else:
        plot_generic_node_panel(node, x_batch, save_path)

def plot_pareto_front(csv_path, save_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # ----------------------------------------------------
    # Load data
    # ----------------------------------------------------
    df = pd.read_csv(csv_path)

    df = df[['Model', 'Dataset', 'CLIoU', 'nParams']]
    df = df.dropna(subset=['CLIoU', 'nParams'])

    df['nParams'] = pd.to_numeric(df['nParams'])
    df['CLIoU']   = pd.to_numeric(df['CLIoU'])

    # ----------------------------------------------------
    # COLOR SETUP BY DATASET
    # ----------------------------------------------------
    datasets = df['Dataset'].unique()
    colors   = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    color_map = {ds: colors[i] for i, ds in enumerate(datasets)}

    # Different markers for models
    ModelMarkers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*']
    model_list = df['Model'].unique()
    model_markers = {m: ModelMarkers[i % len(ModelMarkers)] for i, m in enumerate(model_list)}

    # ----------------------------------------------------
    # PLOT 1: Pareto scatter plot (NO LINES)
    # ----------------------------------------------------
    plt.figure(figsize=(6, 4))

    # ---- Dataset-wise scatter ----
    for ds in datasets:
        subset = df[df['Dataset'] == ds]

        plt.scatter(
            subset['nParams'], subset['CLIoU'],
            color=color_map[ds], s=80, alpha=0.85
        )

        # Text labels
        for _, r in subset.iterrows():
            plt.text(
                r['nParams'], r['CLIoU'],
                r['Model'], fontsize=8,
                color=color_map[ds], alpha=0.8
            )

    # ---- Dataset-wise Pareto points (dots only) ----
    for ds in datasets:
        subset = df[df['Dataset'] == ds].sort_values("nParams")

        pareto = []
        best_ciou = -np.inf

        for _, row in subset.iterrows():
            if row['CLIoU'] > best_ciou:
                pareto.append(row)
                best_ciou = row['CLIoU']

        pareto_df = pd.DataFrame(pareto)

        plt.scatter(
            pareto_df['nParams'], pareto_df['CLIoU'],
            color=color_map[ds], edgecolor="black",
            s=150, marker="o", linewidth=1.5
        )

    plt.xscale("log")
    plt.xlabel("Number of Parameters (log-scale)", fontsize=13)
    plt.ylabel("CLIoU", fontsize=13)
    plt.title("Pareto Scatter Plot per Dataset (No Lines)", fontsize=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(datasets, title="Dataset")
    plt.savefig(save_path.replace(".pdf", "_pareto_front.pdf"), bbox_inches='tight')

    
def plot_efficiency_barplot(csv_path, save_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data
    df = pd.read_csv(csv_path)

    df = df[['Model', 'Dataset', 'CLIoU', 'nParams']]
    df = df.dropna(subset=['CLIoU', 'nParams'])

    # Compute efficiency
    df['Efficiency'] = df['CLIoU'] / (df['nParams'] / 1e6)

    # Seaborn style
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))

    barplot = sns.barplot(
        data=df,
        x="Dataset",
        y="Efficiency",
        hue="Model",
        errorbar=None,
        palette="tab10"
    )

    # Apply LOG SCALE
    plt.yscale("log")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("CLIoU per Million Parameters (log scale)", fontsize=13)
    plt.xlabel("Dataset", fontsize=13)
    plt.title("Model Efficiency per Dataset (Log-Scale)", fontsize=15)

    # Legend on the right
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()





# =========================================================
# Master function to generate all plots for a PCNet / MultiPCNet
# =========================================================
def generate_all_pcnet_plots(model, x, save_root="pcnet_plots"):
    """
    Completely redesigned visualization pipeline:
    - For PCNet: produce node-level explainability panels.
    - For MultiPCNet: recurse into experts + selector.
    """

    os.makedirs(save_root, exist_ok=True)
    # Normalize x to (1,C,H,W)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.ndim == 3:
        x = x.unsqueeze(0)

    plot_class_loglik_maps(model, x, save_dir=save_root)
    plot_node_type_distribution(model, save_dir=save_root)
    plot_depth_histogram(model, save_dir=save_root)
    plot_inputnode_parameter_distributions(model, save_dir=save_root)
    plot_gate_residual_maps(model, x, save_dir=save_root)
    plot_input_saliencies(model, x, save_dir=save_root)

    # -------------------------------
    # MultiPCNet
    # -------------------------------
    
    if hasattr(model, "n_experts"):
        for i, expert in enumerate(model.experts):
            subdir = os.path.join(save_root, f"expert_{i}")
            os.makedirs(subdir, exist_ok=True)
            generate_all_pcnet_plots(expert.model, x, save_root=subdir)
        return

    # -------------------------------
    # PCNet: visualize each node
    # -------------------------------
    nodes = list(model.get_nodes())
    node_dir = os.path.join(save_root, "nodes")
    os.makedirs(node_dir, exist_ok=True)

    for i, node in enumerate(nodes):
        visualize_node(node, x, node_dir, i)
    
    # -------------------------------
    # PCNet: visualize flow of influence
    # -------------------------------
    plot_pcnet_topology(model, save_path=os.path.join(save_root, "pcnet_topology.pdf"))
    plot_flow_influence(model, x, save_path=os.path.join(save_root, "pc_flow.pdf"))
    plot_distribution_flow_paths(model, x,
                                 save_path=os.path.join(save_root, "distribution_paths.pdf"))


    print(f"[OK] Saved all PCNet visualizations to {save_root}")

if __name__ == "__main__":
    # image_path = "data/DeepCrack/test/images/"
    # mask_path = "data/DeepCrack/test/masks/"
    # data = CrackDataset(image_path, mask_path, resize_size=(400, 400))
    # m = pickle.load(open("logs/old/pcnet_ge_small/DeepCrack/1/pcnet-ge.pkl", "rb"))
    # print(m)
    # # m = MultiPCNet(base_pcnet_cls=PCNet, n_experts=5)

    # x = torch.stack([data[i][0] for i in range(3)], dim=0)

    # generate_all_pcnet_plots(m, x, save_root="plots")

    pareto_file = "logs/Paretofront.csv"
    plot_pareto_front(pareto_file, save_path="plots/pareto_front.pdf")
    plot_efficiency_barplot(pareto_file, save_path="plots/efficiency_barplot.pdf")
    # Comparison between the models
    