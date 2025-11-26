import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from probabilistic_circuits.probabilistic_circuits import *
from crackdata_processing import *
import pickle

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
    plt.title("PCNet / MultiPCNet Topology", fontsize=16)
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
        plt.suptitle("MultiPCNet Flow Influence", fontsize=18)

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

    ax.set_title(title, fontsize=12)
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
def plot_generic_node_panel(node, x, save_path):
    act = node.evaluate(x).detach().cpu().numpy()
    act = _to_spatial_map(act)

    lo, hi = np.percentile(act, [5, 95])
    act_norm = np.clip((act - lo) / (hi - lo + 1e-8), 0, 1)

    # Input image for overlay
    img = x[0].detach().cpu().numpy().transpose(1, 2, 0)
    if img.shape[-1] == 1:
        img = img[:, :, 0]

    # overlay
    overlay = (0.6 * img + 0.4 * act_norm[..., None]).clip(0, 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 1. original
    axs[0, 0].imshow(img, cmap="gray")
    axs[0, 0].set_title("Original image")
    axs[0, 0].axis("off")

    # 2. heatmap
    im = axs[0, 1].imshow(act, cmap="viridis")
    axs[0, 1].set_title(f"{node.__class__.__name__} activation")
    axs[0, 1].axis("off")
    fig.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # 3. overlay
    axs[1, 0].imshow(overlay)
    axs[1, 0].set_title("Overlay")
    axs[1, 0].axis("off")

    # 4. metadata
    text = [
        f"Node: {node.__class__.__name__}",
        f"min: {act.min():.4f}",
        f"max: {act.max():.4f}",
        f"p5–p95: {lo:.3f} → {hi:.3f}",
    ]
    if hasattr(node, "alpha"):
        text.append(f"Gate alpha = {float(torch.sigmoid(node.alpha)):.3f}")
    if hasattr(node, "beta"):
        text.append(f"Residual beta = {float(torch.sigmoid(node.beta)):.3f}")

    axs[1, 1].text(0, 1, "\n".join(text), va="top", fontsize=12)
    axs[1, 1].axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =========================================================
# PLOTTER: GateNode specialist (shows both children)
# =========================================================
def plot_gate_node(node, x, save_path):
    # gate prob
    gate_val = float(torch.sigmoid(node.alpha))

    # children activations
    acts = []
    for child in node.children:
        a = child.evaluate(x).detach().cpu().numpy()
        acts.append(_to_spatial_map(a))

    left, right = acts

    # normalize both the same way
    lo = min(np.percentile(left, 5), np.percentile(right, 5))
    hi = max(np.percentile(left, 95), np.percentile(right, 95))

    l_norm = np.clip((left - lo) / (hi - lo + 1e-8), 0, 1)
    r_norm = np.clip((right - lo) / (hi - lo + 1e-8), 0, 1)
    diff = l_norm - r_norm

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(left, cmap="viridis")
    axs[0, 0].set_title("Left child log-likelihood")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(right, cmap="viridis")
    axs[0, 1].set_title("Right child log-likelihood")
    axs[0, 1].axis("off")

    im = axs[1, 0].imshow(diff, cmap="coolwarm")
    axs[1, 0].set_title("Difference (Left − Right)")
    axs[1, 0].axis("off")
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

    axs[1, 1].text(
        0, 1,
        f"GateNode\n"
        f"alpha (sigmoid) = {gate_val:.3f}\n"
        f"Left stronger = red\n"
        f"Right stronger = blue\n",
        fontsize=12, va="top"
    )
    axs[1, 1].axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =========================================================
# PLOTTER: SumNode specialist
# =========================================================
def plot_sum_node(node, x, save_path):
    childs = node.children
    K = len(childs)

    # compute all activations
    maps = []
    for ch in childs:
        a = ch.evaluate(x).detach().cpu().numpy()
        maps.append(_to_spatial_map(a))

    # mixture weights
    w = torch.softmax(node.weights, dim=0).detach().cpu().numpy()

    # weighted sum map
    ws_map = sum(w[i] * maps[i] for i in range(K))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # child contributions
    im0 = axs[0, 0].imshow(maps[0], cmap="viridis")
    axs[0, 0].set_title(f"Child 0 (w={w[0]:.2f})")
    axs[0, 0].axis("off")

    if K > 1:
        im1 = axs[0, 1].imshow(maps[1], cmap="viridis")
        axs[0, 1].set_title(f"Child 1 (w={w[1]:.2f})")
        axs[0, 1].axis("off")

    # weighted sum map
    im2 = axs[1, 0].imshow(ws_map, cmap="magma")
    axs[1, 0].set_title("Weighted mixture output")
    axs[1, 0].axis("off")
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # info panel
    axs[1, 1].text(
        0, 1,
        f"SumNode\nChildren: {K}\nWeights:\n{w}",
        fontsize=12, va="top"
    )
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# MASTER DISPATCHER
# =========================================================
def visualize_node(node, x, folder, idx):
    node_type = node.__class__.__name__

    save_path = os.path.join(folder, f"node_{idx}_{node_type}.pdf")

    if node_type == "GateNode":
        plot_gate_node(node, x, save_path)
    elif node_type == "SumNode":
        plot_sum_node(node, x, save_path)
    else:
        plot_generic_node_panel(node, x, save_path)


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


    print(f"[OK] Saved all PCNet visualizations to {save_root}")

if __name__ == "__main__":
    image_path = "data/DeepCrack/test/images/"
    mask_path = "data/DeepCrack/test/masks/"
    data = CrackDataset(image_path, mask_path, resize_size=(400, 400))
    m = pickle.load(open("logs/pcnet_ge/CRACK500/3/pcnet-ge.pkl", "rb"))
    print(m)
    # m = MultiPCNet(base_pcnet_cls=PCNet, n_experts=5)


    generate_all_pcnet_plots(m, data[50][0], save_root="plots_multipcnet")
