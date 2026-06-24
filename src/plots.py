import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
import cv2
import torch.nn.functional as F
import pickle
from crackdata_processing import CrackDataset

plt.style.use('science')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# =========================================================
# HELPER: GRAPH EXTRACTION FROM PYTORCH MODULES
# =========================================================
def extract_pcnet_graph(pcnet):
    """
    Traverses the PCNet starting from pcnet.root using self.child_nodes
    Returns (list_of_nodes, list_of_edges).
    """
    nodes = set()
    edges = set()
    
    if hasattr(pcnet, 'root') and pcnet.root is not None:
        queue = [pcnet.root]
        while queue:
            curr = queue.pop(0)
            if curr not in nodes:
                nodes.add(curr)
                if hasattr(curr, 'child_nodes'):
                    for child in curr.child_nodes:
                        edges.add((curr, child)) # parent, child
                        queue.append(child)
    
    # Failsafe: Add leaves if they are disconnected from the root for some reason
    if hasattr(pcnet, 'leaves'):
        for leaf in pcnet.leaves:
            nodes.add(leaf)
            
    return list(nodes), list(edges)

# =========================================================
# UTIL: convert any activation to a valid (H, W) map
# =========================================================
def _to_spatial_map(arr):
    arr = np.array(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.mean(axis=0)
    if arr.ndim == 1:
        HW = arr.shape[0]
        H = W = int(np.sqrt(HW))
        return arr.reshape(H, W)

    H, W = arr.shape[-2], arr.shape[-1]
    return arr.reshape(-1, H, W).mean(axis=0)

# =========================================================
# 1. STRUCTURAL VISUALIZATIONS
# =========================================================

def safe_graphviz_layout(G, prog="dot"):
    G.graph["graph"] = {
        "rankdir": "TB", "ranksep": "0.1", "nodesep": "0.1",
        "margin": "0", "splines": "ortho",
    }
    G.graph["node"] = {"width": "0.5", "height": "0.5", "shape": "box", "fontsize": "15"}
    G.graph["edge"] = {"minlen": "1", "maxlen": "1.5"}

    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog=prog)
    except Exception:
        pass
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(G, prog=prog)
    except Exception:
        return nx.spring_layout(G)


def plot_pcnet_topology(model, save_path="pcnet_topology.pdf"):
    def add_subgraph(G, pcnet, prefix=""):
        color_map = {
            "InputNode": "#a6cee3",
            "SumNode": "#1f78b4",
            "ProductNode": "#b2df8a",
            "ClassifierNode": "#e31a1c",
            "AttentionalSumNode": "#ff7f00",
            "MultiHeadAttentionalSumNode": "#6a3d9a"
        }

        label_map = {
            "SumNode": r"$\Sigma$",
            "ProductNode": r"$\Pi$",
            "ClassifierNode": r"$C$",
            "AttentionalSumNode": r"$\Sigma_{attn}$",
            "MultiHeadAttentionalSumNode": r"$\Sigma_{mha}$"
        }

        input_counter = 1
        nodes, edges = extract_pcnet_graph(pcnet)

        for n in nodes:
            t = n.__class__.__name__
            node_id = prefix + str(id(n))
            
            if t == "InputNode":
                label = r"$X_{" + str(input_counter) + r"}$"
                input_counter += 1
            else:
                label = label_map.get(t, t)

            G.add_node(node_id, label=label, color=color_map.get(t, "#d3d3d3"))

        for parent, child in edges:
            G.add_edge(prefix + str(id(child)), prefix + str(id(parent)))
            
    G = nx.DiGraph()

    if hasattr(model, "models"): # Random Ensemble Wrapper Check
        for i, e in enumerate(model.models):
            add_subgraph(G, e, prefix=f"exp{i}_")
    else:
        add_subgraph(G, model, prefix="pc_")

    pos = safe_graphviz_layout(G, prog="dot")
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes}

    plt.figure(figsize=(14, 12))
    nx.draw(
        G, pos, labels=node_labels, node_color=node_colors,
        node_size=15000, font_size=50, font_weight="bold",
        edge_color="gray", arrowsize=25,
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def compute_node_activation(pcnet, x):
    acts = {}
    nodes, _ = extract_pcnet_graph(pcnet)
    with torch.no_grad():
        for n in nodes:
            try:
                a = n(x).cpu().numpy()
                acts[n] = _to_spatial_map(a)
            except:
                pass
    return acts


def compute_flow_scores(pcnet, x):
    acts = compute_node_activation(pcnet, x)
    node_score = {n: float(np.mean(np.abs(a))) for n, a in acts.items()}

    _, edges = extract_pcnet_graph(pcnet)
    edge_score = {}
    for parent, child in edges:
        ps = node_score.get(parent, 1e-6)
        cs = node_score.get(child, 1e-6)
        edge_score[(parent, child)] = float(cs / ps)

    return node_score, edge_score


def plot_flow_influence(model, x, save_path="pc_flow.pdf"):
    plt.figure(figsize=(12, 10))
    node_score, edge_score = compute_flow_scores(model, x)

    G = nx.DiGraph()
    nodes, edges = extract_pcnet_graph(model)
    
    for n in nodes:
        G.add_node(str(id(n)), score=node_score.get(n, 0))
    for parent, child in edges:
        G.add_edge(str(id(child)), str(id(parent)), flow=edge_score.get((parent, child), 0))

    pos = safe_graphviz_layout(G, prog="dot")

    node_vals = np.array([G.nodes[n]["score"] for n in G.nodes])
    node_min, node_max = node_vals.min(), node_vals.max()
    node_norm = (node_vals - node_min) / (node_max - node_min + 1e-8)

    edge_vals = np.array([G.edges[e]["flow"] for e in G.edges])
    e_min, e_max = edge_vals.min(), edge_vals.max()
    edge_norm = (edge_vals - e_min) / (e_max - e_min + 1e-8)

    edge_widths = 0.5 + 4.0 * edge_norm
    edge_colors = plt.cm.plasma(edge_norm)
    
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, node_color=node_norm, cmap="plasma", node_size=2400, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowsize=18, ax=ax)
    
    ax.set_title("PCNet Flow Influence")
    ax.axis("off")

    sm_nodes = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(node_min, node_max))
    cbar1 = plt.colorbar(sm_nodes, ax=ax, fraction=0.04, pad=0.02)
    cbar1.set_label("Node importance", fontsize=8)

    sm_edges = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(e_min, e_max))
    cbar2 = plt.colorbar(sm_edges, ax=ax, fraction=0.04, pad=0.10)
    cbar2.set_label("Edge flow", fontsize=8)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_node_type_distribution(pcnet, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    nodes, _ = extract_pcnet_graph(pcnet)
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
    os.makedirs(save_dir, exist_ok=True)
    if not hasattr(pcnet, 'root') or pcnet.root is None:
        return
        
    depths = {pcnet.root: 0}
    queue = [pcnet.root]

    while queue:
        n = queue.pop(0)
        if hasattr(n, 'child_nodes'):
            for ch in n.child_nodes:
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
    nodes, _ = extract_pcnet_graph(pcnet)

    mus, sigmas, nus = [], [], []
    for n in nodes:
        if n.__class__.__name__ == "InputNode":
            mus.append(n.mu.item())
            sigmas.append(torch.exp(n.log_sigma).item())
            nus.append(torch.exp(n.log_nu).item())

    if not mus: return # Skip if no input nodes found
    
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].hist(mus, bins=20); axs[0].set_title("mu distribution")
    axs[1].hist(sigmas, bins=20); axs[1].set_title("sigma distribution")
    axs[2].hist(nus, bins=20); axs[2].set_title("nu distribution")
    fig.tight_layout()
    plt.savefig(f"{save_dir}/inputnode_param_distributions.pdf")
    plt.close()


def plot_gate_residual_maps(pcnet, x, save_dir):
    """ Gate values are now properties inside InputNode """
    os.makedirs(save_dir, exist_ok=True)
    nodes, _ = extract_pcnet_graph(pcnet)

    gate_vals = []
    for n in nodes:
        if n.__class__.__name__ == "InputNode" and hasattr(n, 'gate'):
            gate_vals.append(torch.sigmoid(n.gate).item())

    if gate_vals:
        plt.figure(figsize=(6,4))
        plt.hist(gate_vals, bins=20)
        plt.title("InputNode Gate alpha values (sigmoid(gate))")
        plt.savefig(f"{save_dir}/gate_distributions.pdf", bbox_inches='tight')
        plt.close()

# =========================================================
# 2. ACTIVATION & SALIENCY MAPS
# =========================================================

def plot_class_loglik_maps(pcnet, x, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        logits = pcnet(x)

    ll = logits[0].cpu().numpy()
    if ll.ndim == 1: ll = ll[:, None, None] # Pad if scalar
    C = ll.shape[0]

    vmin, vmax = np.percentile(ll, 5), np.percentile(ll, 95)

    fig, axs = plt.subplots(1, C, figsize=(6*C, 5))
    if C == 1: axs = [axs]

    for c in range(C):
        im = axs[c].imshow(ll[c], cmap="magma", vmin=vmin, vmax=vmax)
        axs[c].set_title(f"Class {c} log-likelihood")
        axs[c].axis("off")
        fig.colorbar(im, ax=axs[c], fraction=0.045, pad=0.03)
        
    plt.savefig(f"{save_dir}/class_loglik_maps.pdf", bbox_inches='tight')
    plt.close()


def plot_input_saliencies(pcnet, x, save_dir, n_rows=3, vmax_percentile=99.0, use_log=True):
    os.makedirs(save_dir, exist_ok=True)

    def _to_final_class_mask(a_np):
        a = np.asarray(a_np)
        if a.ndim >= 1 and a.shape[0] == 1: a = np.squeeze(a, axis=0)
        if a.ndim == 3 and a.shape[0] > 1:
            cls = np.argmax(a, axis=0)
            return (~(cls == 1)).astype(np.float32)
        if a.ndim == 3 and a.shape[0] == 1: a = a[0]
        if a.ndim == 2: return (~(a > 0.5)).astype(np.float32)
        return np.ones((1, 1), dtype=np.float32)

    x = x.clone().detach().requires_grad_(True)
    out = pcnet(x)
    out.sum().backward()

    grad = x.grad.detach().cpu().numpy()
    inp = x.detach().cpu().numpy()

    with torch.no_grad():
        out_pred = pcnet(x.detach()).cpu().numpy()

    B, C, H, W = grad.shape
    R = min(n_rows, B)
    sal = np.abs(grad[:R])
    if use_log: sal = np.log1p(sal)

    vmax, vmin = np.percentile(sal, vmax_percentile), 0.0
    final_bg_masks = np.stack([_to_final_class_mask(out_pred[r:r+1]) for r in range(R)], axis=0)

    n_cols = C + 2
    fig, axs = plt.subplots(R, n_cols, figsize=(3 * n_cols, 3 * R))
    if R == 1: axs = axs.reshape(1, n_cols)
    fig.subplots_adjust(right=0.92)

    last_im = None
    for r in range(R):
        img = inp[r]
        if img.shape[0] == 1: axs[r, 0].imshow(img[0], cmap="gray")
        else: axs[r, 0].imshow(np.transpose(img, (1, 2, 0)))
        axs[r, 0].axis("off")
        axs[r, 0].set_ylabel(f"Image {r}", fontsize=20)

        for c in range(C):
            last_im = axs[r, c + 1].imshow(sal[r, c], cmap="inferno", vmin=vmin, vmax=vmax)
            axs[r, c + 1].axis("off")

        axs[r, -1].imshow(final_bg_masks[r], cmap="gray", vmin=0, vmax=1)
        axs[r, -1].axis("off")

    axs[0, 0].set_title("Input", fontsize=20)
    for c in range(C): axs[0, c + 1].set_title(f"Saliency ch {c}", fontsize=20)
    axs[0, -1].set_title("Classification", fontsize=20)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.945, 0.15, 0.010, 0.7])
        fig.colorbar(last_im, cax=cbar_ax).set_label(r"$|\partial \log p(x) / \partial x|$", fontsize=20)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.savefig(os.path.join(save_dir, "input_saliencies_segmentation.pdf"), dpi=300)
    plt.close()


def plot_generic_node_panel(node, x_batch, save_path):
    x_batch = x_batch.detach().cpu()
    N = x_batch.shape[0]
    acts = []
    
    with torch.no_grad():
        for i in range(N):
            a = node(x_batch[i:i+1]).cpu().numpy()
            acts.append(_to_spatial_map(a))

    fig, axs = plt.subplots(N, 3, figsize=(12, 4*N))
    if N == 1: axs = axs.reshape(1, 3)

    for i in range(N):
        img = x_batch[i].numpy().transpose(1, 2, 0)
        if img.shape[-1] == 1: img = img[:, :, 0]

        act = acts[i]
        lo, hi = np.percentile(act, [5, 95])
        act_norm = np.clip((act - lo) / (hi - lo + 1e-8), 0, 1)
        overlay = (0.6 * img + 0.4 * act_norm[..., None] if img.ndim==3 else 0.4*act_norm).clip(0, 1)

        axs[i, 0].imshow(img, cmap="gray")
        axs[i, 0].set_title(f"Original {i}")
        axs[i, 0].axis("off")

        im = axs[i, 1].imshow(act, cmap="plasma")
        axs[i, 1].set_title(f"{node.__class__.__name__} activation {i}")
        axs[i, 1].axis("off")
        fig.colorbar(im, ax=axs[i, 1], fraction=0.05, pad=0.04)

        axs[i, 2].imshow(overlay)
        axs[i, 2].set_title(f"Overlay {i}")
        axs[i, 2].axis("off")
        
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_sum_node(node, x_batch, save_path):
    N = x_batch.shape[0]
    # Updated to use child_nodes
    K = len(node.child_nodes)
    maps = [ [] for _ in range(K) ]

    with torch.no_grad():
        for i in range(N):
            for k, child in enumerate(node.child_nodes):
                a = child(x_batch[i:i+1]).cpu().numpy()
                maps[k].append(_to_spatial_map(a))

    # Support dynamically calculated attention weights if available, else static
    if hasattr(node, 'weights'):
        w = torch.softmax(node.weights, dim=0).detach().cpu().numpy()
    else:
        w = np.ones(K) / K 

    fig, axs = plt.subplots(N, 4, figsize=(16, 4*N))
    if N == 1: axs = axs.reshape(1,4)

    for i in range(N):
        if K > 0:
            axs[i,0].imshow(maps[0][i], cmap="plasma")
            axs[i,0].set_title(f"Child 0 (w={w[0]:.2f})")
        axs[i,0].axis("off")

        if K > 1:
            axs[i,1].imshow(maps[1][i], cmap="plasma")
            axs[i,1].set_title(f"Child 1 (w={w[1]:.2f})")
        axs[i,1].axis("off")

        if K > 1:
            mix = w[0]*maps[0][i] + w[1]*maps[1][i]
            im = axs[i,2].imshow(mix, cmap="magma")
            axs[i,2].set_title("Weighted output")
            fig.colorbar(im, ax=axs[i,2], fraction=0.05, pad=0.04)
        axs[i,2].axis("off")

        axs[i,3].text(0,1, f"Node: {node.__class__.__name__}\nChildren={K}\nWeights={w}", fontsize=10, va="top")
        axs[i,3].axis("off")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# =========================================================
# MASTER DISPATCHER
# =========================================================
def visualize_node(node, x_batch, folder, idx):
    node_type = node.__class__.__name__
    save_path = os.path.join(folder, f"node_{idx}_{node_type}.pdf")

    if "SumNode" in node_type:
        plot_sum_node(node, x_batch, save_path)
    else:
        plot_generic_node_panel(node, x_batch, save_path)

def generate_all_pcnet_plots(model, x, save_root="pcnet_plots"):
    os.makedirs(save_root, exist_ok=True)
    if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
    if x.ndim == 3: x = x.unsqueeze(0)

    plot_class_loglik_maps(model, x, save_dir=save_root)
    plot_node_type_distribution(model, save_dir=save_root)
    plot_depth_histogram(model, save_dir=save_root)
    plot_inputnode_parameter_distributions(model, save_dir=save_root)
    plot_gate_residual_maps(model, x, save_dir=save_root)
    plot_input_saliencies(model, x, save_dir=save_root)

    # MultiPCNet / Random Ensemble compatibility
    if hasattr(model, "models"): 
        for i, expert in enumerate(model.models):
            subdir = os.path.join(save_root, f"expert_{i}")
            generate_all_pcnet_plots(expert, x, save_root=subdir)
        return

    nodes, _ = extract_pcnet_graph(model)
    node_dir = os.path.join(save_root, "nodes")
    os.makedirs(node_dir, exist_ok=True)

    for i, node in enumerate(nodes):
        visualize_node(node, x, node_dir, i)
    
    plot_pcnet_topology(model, save_path=os.path.join(save_root, "pcnet_topology.pdf"))
    plot_flow_influence(model, x, save_path=os.path.join(save_root, "pc_flow.pdf"))

    print(f"[OK] Saved all PCNet visualizations to {save_root}")


if __name__ == "__main__":

    import os
    import random
    import matplotlib.pyplot as plt
    from PIL import Image

    # # Root directory containing all datasets
    # ROOT_DIR = "data/crackseg9k_split"

    # # Dataset names in required order
    # DATASETS = ["AEL", "Crack500", "DeepCrack", "GAPS384", "cracktree200", "CrackSeg9k"]

    # NUM_IMAGES = 4  # images per dataset

    # fig, axes = plt.subplots(
    #     nrows=len(DATASETS),
    #     ncols=NUM_IMAGES,
    #     figsize=(10, 12), 
    #     gridspec_kw={"wspace": 0.02, "hspace": 0.02}
    # )

    # for row, dataset in enumerate(DATASETS):
    #     images_dir = os.path.join(ROOT_DIR, dataset, "train", "images")
        
    #     # Get image files
    #     try:
    #         image_files = [
    #             f for f in os.listdir(images_dir)
    #             if f.lower().endswith((".png", ".jpg", ".jpeg"))
    #         ]
    #         selected_images = random.sample(image_files, NUM_IMAGES)
    #     except (FileNotFoundError, ValueError):
    #         # Fallback if directory is missing or empty
    #         selected_images = [None] * NUM_IMAGES

    #     for col, img_name in enumerate(selected_images):
    #         ax = axes[row, col]
            
    #         if img_name:
    #             img_path = os.path.join(images_dir, img_name)
    #             img = Image.open(img_path).resize((512, 512))
    #             ax.imshow(img)
            
    #         # Remove ticks and spines for all images
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         for spine in ax.spines.values():
    #             spine.set_visible(False)

    #         # Apply label ONLY to the first column
    #         if col == 0:
    #             ax.set_ylabel(dataset, fontsize=14, fontweight='bold', rotation=90)
    #             # Center the label vertically (0.5) and push it left (-0.1)
    #             ax.yaxis.set_label_coords(-0.15, 0.5)
    #             # Ensure the y-label is actually rendered
    #             ax.yaxis.set_visible(True)

    # # Adjust layout to make room for labels on the left
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.1) 

    # # Save the result
    # os.makedirs("plots", exist_ok=True)
    # plt.savefig("plots/dataset_samples.pdf", bbox_inches="tight")
    # exit()
    root = "logs/pcnet_ge/100"
    # 
    # plot_evolutionary_results(root)


    image_path = "data/crackseg9k_split/DeepCrack/test/images/"
    mask_path = "data/crackseg9k_split/DeepCrack/test/masks/"
    data = CrackDataset(image_path, mask_path, resize_size=(256, 256))
    m = pickle.load(open("logs/multihead_attentional_pcnet/DeepCrack/5/multihead_attentional_pcnet.pkl", "rb"))
    print(m)
    # m = MultiPCNet(base_pcnet_cls=PCNet, n_experts=5)
    rand_img = np.random.randint(0, len(data), 3)
    x = torch.stack([data[i][0] for i in rand_img], dim=0)

    generate_all_pcnet_plots(m, x, save_root="plots/multihead_attentional_pcnet_deepcrack")

    pareto_file = "logs/Paretofront.csv"
    # plot_pareto_front(pareto_file, save_path="plots/pareto_front_symbols.pdf")
    # plot_efficiency_barplot(pareto_file, save_path="plots/efficiency_barplot.pdf")
    # Comparison between the models
    # models_dict = {
    #     "AEL":{
    #         "model":"logs/pcnet_ge/100/AEL/3/pcnet-ge.pkl",
    #         "image_path":"data/crackseg9k_split/AEL/test/images/",
    #         "mask_path":"data/crackseg9k_split/AEL/test/masks/"
    #     },
    #     "Crack500":{
    #         "model":"logs/pcnet_ge/100/Crack500/1/pcnet-ge.pkl",
    #         "image_path":"data/crackseg9k_split/Crack500/test/images/",
    #         "mask_path":"data/crackseg9k_split/Crack500/test/masks/"
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