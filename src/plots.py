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

class PCNetAnalyzer:
    def __init__(self, pcnet):
        self.pcnet = pcnet

    # --------------------------
    # Histogram of gate / residual coefficients
    # --------------------------
    def plot_parameter_histograms(self, save_path=None):
        gate_vals, res_vals = [], []
        for n in self.pcnet.get_nodes():
            if isinstance(n, GateNode):
                gate_vals.append(torch.sigmoid(n.alpha).item())
            elif isinstance(n, ResidualNode):
                res_vals.append(torch.sigmoid(n.beta).item())

        plt.figure(figsize=(10,4))
        if gate_vals:
            sns.histplot(gate_vals, kde=True, color="orange", label="GateNode α (sigmoid)")
        if res_vals:
            sns.histplot(res_vals, kde=True, color="skyblue", label="ResidualNode β (sigmoid)")
        plt.legend()
        plt.title("Distribution of Gate / Residual Coefficients")
        plt.xlabel("Value (after sigmoid)")
        plt.ylabel("Count")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved histogram to {save_path}")
        plt.show()

    # --------------------------
    # Heatmap visualization of per-feature gate/residual activation
    # --------------------------
    def activation_heatmap(self, inputs, save_path=None):
        """
        Compute per-feature average gate and residual activation strength.
        Args:
            inputs (torch.Tensor): input batch (B, C, H, W)
        """
        activations = []
        with torch.no_grad():
            for node in self.pcnet.get_nodes():
                if isinstance(node, GateNode):
                    gate_val = torch.sigmoid(node.alpha).item()
                    activations.append(("GateNode", node, gate_val))
                elif isinstance(node, ResidualNode):
                    res_val = torch.sigmoid(node.beta).item()
                    activations.append(("ResidualNode", node, res_val))

        if not activations:
            print("No Gate/Residual nodes to visualize.")
            return

        # Convert to heatmap-like tensor for visualization
        names, vals = zip(*[(a[0], a[2]) for a in activations])
        vals = np.array(vals)

        # Plot
        plt.figure(figsize=(8, max(2, len(vals)*0.2)))
        sns.heatmap(vals.reshape(-1,1),
                    annot=True, fmt=".2f",
                    cmap="coolwarm", cbar=True,
                    yticklabels=names,
                    xticklabels=["Activation"])
        plt.title("Gate / Residual Activation Heatmap")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        plt.show()

        # ----------------------------------------------------------
    # Per-sample GateNode activation maps (dynamic visualization)
    # ----------------------------------------------------------
    def sample_gate_maps(self, inputs, max_nodes=4, save_dir=None):
        """
        Visualize GateNode activations for specific inputs.
        Each GateNode outputs a gate value per image -> heatmap overlay.

        Args:
            inputs (torch.Tensor): shape (B, C, H, W)
            max_nodes (int): number of GateNodes to visualize
            save_dir (str): directory to save plots (optional)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        gate_nodes = [n for n in self.pcnet.get_nodes() if isinstance(n, GateNode)]
        if not gate_nodes:
            print("No GateNodes found for visualization.")
            return

        gate_nodes = gate_nodes[:max_nodes]

        with torch.no_grad():
            for i, node in enumerate(gate_nodes):
                gate = torch.sigmoid(node.alpha).item()
                # Estimate activation map: combine child log-likelihoods spatially
                A = node.children[0].evaluate(inputs)  # (B, H, W)
                B = node.children[1].evaluate(inputs)
                # Compute soft contribution of each pixel
                contrib = torch.softmax(torch.stack([A, B], dim=0), dim=0)[0]  # select first child weight
                mean_map = contrib.mean(dim=0).cpu().numpy()

                plt.figure(figsize=(4,4))
                sns.heatmap(mean_map, cmap="magma", cbar=False)
                plt.title(f"GateNode {i} — α={gate:.2f}")
                if save_dir:
                    path = os.path.join(save_dir, f"gate_map_{i}.png")
                    plt.savefig(path, bbox_inches='tight')
                    print(f"Saved {path}")
                plt.show()


    # --------------------------
    # Combined summary
    # --------------------------
    def summary(self, inputs=None, save_dir=None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        hist_path = os.path.join(save_dir, "gate_res_hist.png") if save_dir else None
        heat_path = os.path.join(save_dir, "gate_res_heatmap.png") if save_dir else None

        self.plot_parameter_histograms(save_path=hist_path)
        if inputs is not None:
            self.activation_heatmap(inputs, save_path=heat_path)

if __name__ == "__main__":
    path = "data/cracktree200/test"
    images_path = path + "images/"
    ground_truth_path = path + "/masks/"        
    data = CrackDataset(images_path, ground_truth_path,sample_size=4, resize_size=(256,256))
    pcnet = pickle.load(open("logs/pcnet_ge/DeepCrack/1001/pcnet-ge.pkl", "rb"))
    analyzer = PCNetAnalyzer(pcnet)
    analyzer.summary()
    analyzer.sample_gate_maps(data, save_dir="./gate_maps")