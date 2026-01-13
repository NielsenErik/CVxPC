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
        node_size=15000,
        font_size=25,
        font_weight="bold",
        edge_color="gray",
        arrowsize=25,
    )
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
        cmap="plasma",
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
    sm_nodes = plt.cm.ScalarMappable(cmap="plasma",
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

    plt.figure(figsize=(12,8))
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

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_input_saliencies(
    pcnet,
    x,
    save_dir,
    n_rows=3,
    vmax_percentile=99.0,
    use_log=True
):
    """
    Rows  : different images in batch
    Col 0: input image
    Col 1..C: saliency maps per channel
    """
    import torch

    os.makedirs(save_dir, exist_ok=True)

    # -------- Forward & backward --------
    x = x.clone().detach().requires_grad_(True)
    out = pcnet.evaluate(x)
    out.sum().backward()

    grad = x.grad.detach().cpu().numpy()
    inp = x.detach().cpu().numpy()

    # -------- Normalize shapes → (B, C, H, W) --------
    if grad.ndim == 4:
        pass
    elif grad.ndim == 3:
        grad = grad[None, ...]
        inp = inp[None, ...]
    else:
        raise ValueError(f"Unexpected input shape: {grad.shape}")

    B, C, H, W = grad.shape
    R = min(n_rows, B)

    sal = np.abs(grad[:R])

    if use_log:
        sal = np.log1p(sal)

    # Robust shared scale (avoid black rows)
    vmax = np.percentile(sal, vmax_percentile)
    vmin = 0.0

    # -------- Figure layout --------
    n_cols = C + 1  # input + saliency channels
    fig, axs = plt.subplots(
        nrows=R,
        ncols=n_cols,
        figsize=(4 * n_cols, 3.2 * R),
        constrained_layout=True
    )

    # Ensure 2D axes array
    if R == 1:
        axs = axs.reshape(1, n_cols)

    last_im = None

    for r in range(R):

        # ----- Column 0: input image -----
        img = inp[r]

        # If grayscale, collapse channel
        if img.shape[0] == 1:
            img_show = img[0]
            axs[r, 0].imshow(img_show, cmap="gray")
        else:
            # assume CHW → HWC
            img_show = np.transpose(img, (1, 2, 0))
            axs[r, 0].imshow(img_show)

        axs[r, 0].axis("off")

        # ----- Saliency columns -----
        for c in range(C):
            last_im = axs[r, c + 1].imshow(
                sal[r, c],
                cmap="inferno",
                vmin=vmin,
                vmax=vmax
            )
            axs[r, c + 1].axis("off")

        # Row label
        axs[r, 0].set_ylabel(f"Image {r}", fontsize=20)

    # -------- Column titles (first row only) --------
    axs[0, 0].set_title("Input image", fontsize=20)
    for c in range(C):
        axs[0, c + 1].set_title(f"Saliency ch {c}", fontsize=20)

    # -------- Single shared colorbar --------
    cbar = fig.colorbar(
        last_im,
        ax=axs[:, 1:],  # exclude input column
        location="right",
        shrink=0.95,
        pad=0.02
    )
    cbar.set_label(
        r"$|\partial \log p(x) / \partial x|$" + (" (log)" if use_log else ""),
        fontsize=20
    )

    plt.savefig(
        os.path.join(save_dir, "input_plus_saliencies.pdf"),
        bbox_inches="tight"
    )
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
        im = axs[i, 1].imshow(act, cmap="plasma")
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

        axs[i,0].imshow(L, cmap="plasma")
        axs[i,0].set_title(f"Left {i}")
        axs[i,0].axis("off")

        axs[i,1].imshow(R, cmap="plasma")
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
        axs[i,0].imshow(maps[0][i], cmap="plasma")
        axs[i,0].set_title(f"Child 0 (w={w[0]:.2f})")
        axs[i,0].axis("off")

        axs[i,1].imshow(maps[1][i], cmap="plasma")
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
    Shows the distribution evolution for each input image (row).
    Columns follow the PCNet computational path:
    Input → InputNodes → internal nodes → ClassifierNode → FINAL CLASSIFICATION (mask)

    FINAL CLASSIFICATION VIS:
      - white background
      - black crack
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    x_batch = x_batch.detach().cpu()
    B = x_batch.shape[0]

    # ---------------------------------------------------------
    # 1. Get model nodes in topological evaluation order
    # ---------------------------------------------------------
    nodes = list(model.get_nodes())

    # reorder: ensure InputNodes first, ClassifierNodes last
    def order_key(n):
        name = n.__class__.__name__
        if name == "InputNode":
            return 0
        if name == "ClassifierNode":
            return 2
        return 1

    nodes = sorted(nodes, key=order_key)

    # ---------------------------------------------------------
    # Helper: convert raw node output to spatial heatmap
    # ---------------------------------------------------------
    def _to_spatial_map(a_np: np.ndarray) -> np.ndarray:
        """
        Accepts shapes like:
          (1,C,H,W), (C,H,W), (1,H,W), (H,W), (N,) ...
        Returns (H,W) if possible, else (1,1) fallback.
        """
        a = a_np
        # squeeze batch dim if present
        if a.ndim >= 1 and a.shape[0] == 1:
            a = np.squeeze(a, axis=0)

        if a.ndim == 2:  # (H,W)
            return a
        if a.ndim == 3:  # (C,H,W) or (H,W,C)
            # assume channel-first if first dim small-ish
            if a.shape[0] <= 16 and a.shape[1] > 8 and a.shape[2] > 8:
                return a.mean(axis=0)  # (H,W)
            # else channel-last
            if a.shape[2] <= 16 and a.shape[0] > 8 and a.shape[1] > 8:
                return a.mean(axis=2)
            # fallback
            return a[0] if a.shape[0] > 0 else np.zeros((1, 1), dtype=np.float32)
        if a.ndim == 1:
            return np.array([[a.mean()]], dtype=np.float32)
        if a.ndim == 4:  # (B,C,H,W) but batch should already be 1; just in case:
            return a[0].mean(axis=0)
        return np.zeros((1, 1), dtype=np.float32)

    # ---------------------------------------------------------
    # Helper: build final classification mask (white bg, black crack)
    # ---------------------------------------------------------
    def _to_final_class_mask(a_np: np.ndarray) -> np.ndarray:
        """
        Tries to interpret classifier output as:
          - multi-class logits/probs: (1,C,H,W) or (C,H,W) -> argmax
          - binary logits/probs:      (1,1,H,W) or (H,W)  -> threshold
        Returns mask in {0,1} where:
          1 = background (white)
          0 = crack (black)
        """
        a = a_np
        if a.ndim >= 1 and a.shape[0] == 1:
            a = np.squeeze(a, axis=0)  # remove batch if present

        # Multi-class: (C,H,W)
        if a.ndim == 3 and a.shape[0] > 1:
            cls = np.argmax(a, axis=0)  # (H,W)
            # assume "crack" is class 1 if exists; otherwise crack = any non-zero class
            crack = (cls == 1) if a.shape[0] > 1 else (cls != 0)
            bg = (~crack).astype(np.float32)
            return bg  # 1 background, 0 crack

        # Binary: (1,H,W) or (H,W)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 2:
            # if it's logits, threshold at 0; if probs, threshold at 0.5 — we handle both safely:
            # use 0.5 if values look like probs, else 0.0
            finite = a[np.isfinite(a)]
            if finite.size > 0 and finite.min() >= 0.0 and finite.max() <= 1.0:
                crack = a > 0.5
            else:
                crack = a > 0.0
            bg = (~crack).astype(np.float32)
            return bg

        # Scalar fallback
        return np.ones((1, 1), dtype=np.float32)

    # ---------------------------------------------------------
    # 2. Evaluate all nodes for each input image
    # ---------------------------------------------------------
    all_maps = []   # list of rows; each row is list of spatial maps (or None)
    all_last_raw = []  # store raw output of last (ClassifierNode if present) for final mask
    for img_idx in range(B):
        x = x_batch[img_idx:img_idx + 1]

        row_maps = []
        last_raw = None
        for node in nodes:
            try:
                a_t = node.evaluate(x).detach().cpu()
                a_np = a_t.numpy()
                last_raw = a_np  # keep overwriting; ends with last node output
                a_map = _to_spatial_map(a_np)
                row_maps.append(a_map)
            except Exception:
                row_maps.append(None)

        all_maps.append(row_maps)
        all_last_raw.append(last_raw)

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
    #     - columns = Input → Nodes... → FINAL CLASSIFICATION
    # ---------------------------------------------------------
    T = len(nodes)
    fig, axs = plt.subplots(B, T + 2, figsize=(3 * (T + 2), 3 * B))

    if B == 1:
        axs = axs.reshape(1, T + 2)

    # Column 0: original image
    for i in range(B):
        img = x_batch[i].numpy().transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img[:, :, 0]

        axs[i, 0].imshow(img, cmap="gray")
        if i == 0:
            axs[i, 0].set_title("Input", fontsize=20)
        axs[i, 0].axis("off")

    # Node activations (columns 1..T+1)
    im_for_cbar = None
    for row_i in range(B):
        for col_j, node in enumerate(nodes):
            ax = axs[row_i, col_j + 1]
            act = all_maps[row_i][col_j]
            title = f"{col_j}: {node.__class__.__name__}"

            if act is None:
                if row_i == 0:
                    ax.set_title(title, fontsize=20)
                ax.axis("off")
                continue

            im = ax.imshow(act, cmap="plasma", vmin=lo, vmax=hi)
            if im_for_cbar is None:
                im_for_cbar = im
            if row_i == 0:
                ax.set_title(title, fontsize=20)
            ax.axis("off")

    # Final classification (last column, index T+1)
    for row_i in range(B):
        ax = axs[row_i, T + 1]
        raw = all_last_raw[row_i]
        if raw is None:
            ax.set_title("Final\n(no output)", fontsize=20)
            ax.axis("off")
            continue

        bg_mask = _to_final_class_mask(raw)  # 1=white bg, 0=black crack
        ax.imshow(bg_mask, cmap="gray", vmin=0, vmax=1)
        if row_i == 0:
            ax.set_title("Classification", fontsize=20)

        ax.axis("off")

    # Add a single global colorbar for activations only
    if im_for_cbar is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.945, 0.15, 0.010, 0.7])

        fig.colorbar(im_for_cbar, cax=cbar_ax)

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


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

# ----------------------------
# Helper: model output -> crack mask
# ----------------------------
def cl_iou_score(pred_mask, true_mask, tau=4):
    """
    pred_mask, true_mask: (B,1,H,W) in [0,1] (will be thresholded at 0.5 inside)
    """
    device = pred_mask.device
    pred_mask = (pred_mask > 0.5).float()
    true_mask = (true_mask > 0.5).float()

    # Skeletonization approximation
    eroded_true = F.max_pool2d(1 - true_mask, 3, 1, 1)
    eroded_pred = F.max_pool2d(1 - pred_mask, 3, 1, 1)
    skel_true = torch.relu(true_mask - (1 - eroded_true))
    skel_pred = torch.relu(pred_mask - (1 - eroded_pred))

    # --- Dilation kernel (ellipse) ---
    k_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tau + 1, 2 * tau + 1))
    k = torch.tensor(k_np, dtype=torch.float32, device=device)[None, None, :, :]

    dil_true = (F.conv2d(skel_true, k, padding=tau, groups=1) > 0).float()
    dil_pred = (F.conv2d(skel_pred, k, padding=tau, groups=1) > 0).float()

    TPcl = (skel_true * dil_pred).sum(dim=(1, 2, 3))
    FPcl = (skel_pred * (1 - dil_true)).sum(dim=(1, 2, 3))
    FNcl = (skel_true * (1 - dil_pred)).sum(dim=(1, 2, 3))

    cl_iou = TPcl / (TPcl + FPcl + FNcl + 1e-8)
    return cl_iou.mean().item()


# ----------------------------
# Helpers: model output -> prob mask, dataset mask -> tensor
# ----------------------------
def _model_to_pred_prob(out: torch.Tensor) -> torch.Tensor:
    """
    out: (B,C,H,W) logits-like
    returns (B,1,H,W) prob-like in [0,1]
    Assumes class 1 = crack for multiclass.
    """
    if out.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(out.shape)}")
    B, C, H, W = out.shape
    if C == 1:
        return torch.sigmoid(out)
    prob = torch.softmax(out, dim=1)
    return prob[:, 1:2]


def _to_true_prob_mask(y: torch.Tensor) -> torch.Tensor:
    """
    y from CrackDataset: typically (1,H,W) or (H,W).
    returns (1,1,H,W) float in [0,1]
    """
    if y.ndim == 2:
        y = y.unsqueeze(0)  # (1,H,W)
    if y.ndim == 3:
        if y.shape[0] != 1:  # if (C,H,W), take first channel
            y = y[:1]
        y = y.unsqueeze(0)  # (1,1,H,W)
    elif y.ndim == 4:
        # already batched
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(y.shape)}")

    y = y.float()
    # If already {0,1} or [0,1], threshold at 0.5; else treat >0 as crack
    if y.min() >= 0.0 and y.max() <= 1.0:
        y = (y > 0.5).float()
    else:
        y = (y > 0.0).float()
    return y


def _prob_to_bg_vis(prob_fg: torch.Tensor) -> np.ndarray:
    """
    prob_fg: (1,1,H,W) in [0,1]
    returns (H,W) in {0,1} where 1=white background, 0=black crack
    """
    fg = (prob_fg[0, 0] > 0.5).detach().cpu().numpy().astype(np.bool_)
    bg = (~fg).astype(np.float32)
    return bg


# ----------------------------
# Main: evaluate 20, keep top-2 by CL-IoU, plot
# ----------------------------
def plot_top2_by_cliou_per_dataset(models_dict,
                                   save_path="plots/top2_cliou_grid.pdf",
                                   resize_size=(256, 256),
                                   n_test=20,
                                   tau=4,
                                   seed=None,
                                   device=None):
    from crackdata_processing import CrackDataset

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = list(models_dict.keys())
    n_rows = len(datasets)
    n_cols = 6  # (Input, GT, Pred) x 2

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    # column titles only on first row
    col_titles = ["Input (1)", "Mask (1)", "Output (1)",
                  "Input (2)", "Mask (2)", "Output (2)"]
    for j, t in enumerate(col_titles):
        axs[0, j].set_title(t, fontsize=18)

    for r, ds_name in enumerate(datasets):
        entry = models_dict[ds_name]

        # load model
        model = pickle.load(open(entry["model"], "rb"))
        if hasattr(model, "to"):
            model = model.to(device)

        # load dataset
        data = CrackDataset(entry["image_path"], entry["mask_path"], resize_size=resize_size)

        # sample n_test indices (or all if dataset smaller)
        n_pick = min(n_test, len(data))
        idxs = np.random.choice(len(data), size=n_pick, replace=False)

        scored = []
        for idx in idxs:
            x, y = data[idx]
            x_in = x.unsqueeze(0).to(device)  # (1,C,H,W)
            y_true = _to_true_prob_mask(y).to(device)  # (1,1,H,W)

            with torch.no_grad():
                out = model.evaluate(x_in)  # (1,C,H,W)
                pred_prob = _model_to_pred_prob(out)  # (1,1,H,W)

            s = cl_iou_score(pred_prob, y_true, tau=tau)
            scored.append((s, idx, x.detach().cpu(), y.detach().cpu(), pred_prob.detach().cpu()))

        # keep best 2
        scored.sort(key=lambda t: t[0], reverse=True)
        top2 = scored[:2]
        if len(top2) < 2:
            # if dataset has 1 sample, duplicate it (rare)
            top2 = top2 + top2

        # dataset row label (left margin, centered)
        y_center = 1.0 - (r + 0.5) / n_rows
        fig.text(0.015, y_center, ds_name, va="center", ha="left", fontsize=18, rotation=90)

        for k, (s, idx, x_cpu, y_cpu, pred_prob_cpu) in enumerate(top2):
            # input image to numpy
            x_np = x_cpu.numpy()
            img = np.transpose(x_np, (1, 2, 0)) if x_np.ndim == 3 else x_np
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]

            # GT bg mask (white bg, black crack)
            y_true = _to_true_prob_mask(y_cpu)[0, 0].numpy()  # (H,W) in {0,1} where 1=crack
            bg_gt = (1.0 - y_true).astype(np.float32)

            # Pred bg mask (white bg, black crack)
            bg_pred = _prob_to_bg_vis(pred_prob_cpu)

            c0 = 3 * k

            axs[r, c0 + 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axs[r, c0 + 0].axis("off")

            axs[r, c0 + 1].imshow(bg_gt, cmap="gray", vmin=0, vmax=1)
            axs[r, c0 + 1].axis("off")

            axs[r, c0 + 2].imshow(bg_pred, cmap="gray", vmin=0, vmax=1)
            axs[r, c0 + 2].axis("off")


    plt.tight_layout(rect=[0.01, 0.0, 1.0, 1.0])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved -> {save_path}")


# Example call:
# plot_two_examples_per_dataset(models_dict, save_path="plots/datasets_input_gt_pred.pdf", resize_size=(256,256), seed=0)





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

    # import os
    # import random
    # import matplotlib.pyplot as plt
    # from PIL import Image

    # # Root directory containing all datasets
    # ROOT_DIR = "data/crackseg9k_split"

    # # Dataset names in required order
    # DATASETS = ["AEL", "CRACK500", "DeepCrack", "GAPS384", "cracktree200"]

    # NUM_IMAGES = 4  # images per dataset

    # fig, axes = plt.subplots(
    #     nrows=len(DATASETS),
    #     ncols=NUM_IMAGES,
    #     figsize=(8,10),
    #     gridspec_kw={"wspace": 0.05, "hspace": 0.05}
    # )

    # for row, dataset in enumerate(DATASETS):
    #     images_dir = os.path.join(ROOT_DIR, dataset, "train", "images")
        
    #     image_files = [
    #         f for f in os.listdir(images_dir)
    #         if f.lower().endswith((".png", ".jpg", ".jpeg"))
    #     ]

    #     # Randomly select 4 different images
    #     selected_images = random.sample(image_files, NUM_IMAGES)

    #     for col, img_name in enumerate(selected_images):
    #         img_path = os.path.join(images_dir, img_name)
    #         img = Image.open(img_path)
    #         img = img.resize((512, 512))
    #         ax = axes[row, col]
    #         # Set dataset name as y-axis title (only once per row)
    #         ax.imshow(img)
    #         ax.axis("off")
    # for row, dataset in enumerate(DATASETS):
    #     y = 1 - (row + 0.5) / len(DATASETS)
    #     fig.text(
    #         0.045, y, dataset,      # ← CLOSE to images
    #         va="center",
    #         ha="right",
    #         fontsize=13,
    #         rotation=90
    #     )

    # plt.subplots_adjust(
    #     left=0.05,   # ← minimal space, labels still fit
    #     right=0.999,
    #     top=0.98,
    #     bottom=0.05,
    # )

            

    # plt.savefig("plots/dataset_samples.pdf", bbox_inches="tight", pad_inches=0.1)
    # exit()



    image_path = "data/crackseg9k_split/DeepCrack/test/images/"
    mask_path = "data/crackseg9k_split/DeepCrack/test/masks/"
    data = CrackDataset(image_path, mask_path, resize_size=(256, 256))
    m = pickle.load(open("logs/pcnet_ge/100/DeepCrack/3/pcnet-ge.pkl", "rb"))
    print(m)
    # m = MultiPCNet(base_pcnet_cls=PCNet, n_experts=5)
    rand_img = np.random.randint(0, len(data), 3)
    x = torch.stack([data[i][0] for i in rand_img], dim=0)

    generate_all_pcnet_plots(m, x, save_root="plots")

    pareto_file = "logs/Paretofront.csv"
    plot_pareto_front(pareto_file, save_path="plots/pareto_front.pdf")
    plot_efficiency_barplot(pareto_file, save_path="plots/efficiency_barplot.pdf")
    # Comparison between the models
    # models_dict = {
    #     "AEL":{
    #         "model":"logs/pcnet_ge/100/AEL/3/pcnet-ge.pkl",
    #         "image_path":"data/crackseg9k_split/AEL/test/images/",
    #         "mask_path":"data/crackseg9k_split/AEL/test/masks/"
    #     },
    #     "CRACK500":{
    #         "model":"logs/pcnet_ge/100/CRACK500/1/pcnet-ge.pkl",
    #         "image_path":"data/crackseg9k_split/CRACK500/test/images/",
    #         "mask_path":"data/crackseg9k_split/CRACK500/test/masks/"
    #     },
    #     "DeepCrack":{
    #         "model":"logs/pcnet_ge/100/DeepCrack/3/pcnet-ge.pkl",
    #         "image_path":"data/crackseg9k_split/DeepCrack/test/images/",
    #         "mask_path":"data/crackseg9k_split/DeepCrack/test/masks/"
    #     },
    #     "GAPS384":{
    #         "model":"logs/pcnet_ge/100/GAPS384/1/pcnet-ge.pkl",
    #         "image_path":"data/crackseg9k_split/GAPS384/test/images/",
    #         "mask_path":"data/crackseg9k_split/GAPS384/test/masks/"
    #     },
    #     "cracktree200":{
    #         "model":"logs/pcnet_ge/100/cracktree200/3/pcnet-ge.pkl",
    #         "image_path":"data/crackseg9k_split/cracktree200/test/images/",
    #         "mask_path":"data/crackseg9k_split/cracktree200/test/masks/"
    #     },
    #     "CrackSeg9k":{
    #         "model":"logs/pcnet_ge/100/CrackSeg9k/5/pcnet-ge.pkl",
    #         "image_path":"data/CrackSeg9k/test/images/",
    #         "mask_path":"data/CrackSeg9k/test/masks/"
    #     }
    # }
    
    # plot_top2_by_cliou_per_dataset(models_dict, save_path="plots/comparison.pdf", resize_size=(256, 256), n_test=20, tau=4, seed=None, device=None)