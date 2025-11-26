import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from probabilistic_circuits.probabilistic_circuits import *
from crackdata_processing import *
import pickle

import scienceplots
plt.style.use('science')

# =========================================================
# pcnet_plots.py
# Utility to generate structural and interpretability plots
# for PCNet / MultiPCNet models
# =========================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------
# 1. STRUCTURAL VISUALIZATIONS
# ---------------------------------------------------------

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
    plt.ylabel("#Nodes")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/node_type_distribution.png")
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
    plt.ylabel("#Nodes")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/node_depth_histogram.png")
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
    axs[0].hist(mus, bins=20); axs[0].set_title("μ distribution")
    axs[1].hist(sigmas, bins=20); axs[1].set_title("σ distribution")
    axs[2].hist(nus, bins=20); axs[2].set_title("ν distribution")
    fig.tight_layout()
    plt.savefig(f"{save_dir}/inputnode_param_distributions.png")
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


def plot_node_activation_maps(pcnet, x, save_dir, max_nodes=20):
    """
    For each node, save a heatmap of its activation map.
    """
    os.makedirs(save_dir, exist_ok=True)
    node_outputs = eval_all_nodes(pcnet, x)

    for i, (node, act) in enumerate(node_outputs.items()):
        if i >= max_nodes:
            break
        if act.ndim == 4:
            act = act[0,0] if act.shape[1]>1 else act[0,0]   # pick class 0

        plt.figure(figsize=(4,4))
        plt.imshow(act, cmap="viridis")
        plt.colorbar()
        plt.title(node.__class__.__name__)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/node_{i}_{node.__class__.__name__}.png")
        plt.close()


def plot_class_loglik_maps(pcnet, x, save_dir):
    """
    Plot log-likelihood maps for each class.
    """
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        logits = pcnet.evaluate(x)

    logits = logits[0].detach().cpu().numpy()  # (C,H,W)

    C = logits.shape[0]
    fig, axs = plt.subplots(1,C, figsize=(5*C,4))
    for c in range(C):
        axs[c].imshow(logits[c], cmap="magma")
        axs[c].set_title(f"Class {c} log-likelihood")
        axs[c].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_loglik_maps.png")
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
    ax[0].hist(gate_vals); ax[0].set_title("Gate α values (sigmoid(α))")
    ax[1].hist(resid_vals); ax[1].set_title("Residual β values")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gate_residual_distributions.png")
    plt.close()

# ---------------------------------------------------------
# 4. GRADIENT-BASED SALIENCY ("CIRCUIT SALIENCY")
# ---------------------------------------------------------

def plot_input_saliencies(pcnet, x, save_dir):
    """
    Compute ∂ log p(x) / ∂ x for interpretability.
    """
    os.makedirs(save_dir, exist_ok=True)
    x = x.clone().detach().requires_grad_(True)

    out = pcnet.evaluate(x)
    loss = out.sum()
    loss.backward()

    grad = x.grad.detach().cpu().numpy()[0]  # (C,H,W)

    C = grad.shape[0]
    fig, axs = plt.subplots(1,C, figsize=(5*C,4))
    for c in range(C):
        axs[c].imshow(np.abs(grad[c]), cmap="inferno")
        axs[c].set_title(f"Saliency channel {c}")
        axs[c].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/input_saliencies.png")
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
        plt.savefig(f"{save_dir}/selector_weights.png")
        plt.close()

# ---------------------------------------------------------
# MASTER FUNCTION
# ---------------------------------------------------------

def generate_all_pcnet_plots(model, x, save_root="pcnet_plots", max_nodes=20):
    """
    model: PCNet or MultiPCNet
    x: (1,C,H,W) tensor
    """
    os.makedirs(save_root, exist_ok=True)

    if hasattr(model, "experts"):
        # MultiPCNet
        for i, e in enumerate(model.experts):
            subdir = f"{save_root}/expert_{i}"
            generate_all_pcnet_plots(e.model, x, save_root=subdir)

        plot_expert_selection_map(model, x, f"{save_root}/selector")
        return

    # Single PCNet
    plot_node_type_distribution(model, save_root)
    plot_depth_histogram(model, save_root)
    plot_inputnode_parameter_distributions(model, save_root)
    plot_node_activation_maps(model, x, f"{save_root}/node_maps", max_nodes=max_nodes)
    plot_class_loglik_maps(model, x, save_root)
    plot_gate_residual_maps(model, x, save_root)
    plot_input_saliencies(model, x, save_root)

    print(f"All plots saved in: {save_root}")

if __name__ == "__main__":
    image_path = "data/DeepCrack/test/images/"
    mask_path = "data/DeepCrack/test/masks/"
    data = CrackDataset(image_path, mask_path, resize_size=(400, 400))
    m = pickle.load(open("logs/pcnet_ge/DeepCrack/1/pcnet-ge.pkl", "rb"))
    # m = MultiPCNet(base_pcnet_cls=PCNet, n_experts=5)


    generate_all_pcnet_plots(m, data[0][0], save_root="plots_multipcnet")
