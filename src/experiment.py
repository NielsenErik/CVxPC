import mlflow.pytorch
import numpy as np
import cv2
import os
import mlflow
from networks import *
from probabilistic_circuits.probabilistic_circuits import *
from probabilistic_circuits.deep_feature_pcnet import DeepFeaturePCNet, RandomDeepFeaturePCEnsemble
import abc
from torch.utils.data import DataLoader
import tqdm
from tqdm import trange
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
from scipy.stats import norm
from utils import *
import pickle
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import json

import pandas as pd
from crackdata_processing import CrackDataset

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

def average_gradients_across_workers(params):
    """All-reduce gradients in-place and divide by world_size."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


def run_experiment(config):
    if config["algorithm"] == "unet":
        experiment = UNetExperiment(config)
        experiment.run()
    
    elif config["algorithm"] == "unetplus":
        experiment = ExperimentUnetPlus(config)
        experiment.run()
    
    # elif config["algorithm"] == "nnunet":
    #     experiment = ExperimentNNUNet(config)
    #     experiment.run()

    elif config["algorithm"] == "convae":
        experiment = ConvAEExperiment(config)
        experiment.run()

    elif config["algorithm"] == "crackformer":
        experiment = ExperimentCrackFormer(config)
        experiment.run()

    elif config["algorithm"] == "deepcrackz":
        experiment = ExperimentDeepCrackZ(config)
        experiment.run()

    elif config["algorithm"] in ["spatial_pcnet"]:
        experiment = ExperimentSpatialPCNet(config)
        experiment.run()
    
        
    elif config["algorithm"] in ["attentional_pcnet"]:
        experiment = ExperimentAttentionalPCNet(config)
        experiment.run()
        
    elif config["algorithm"] in ["pcnet", "vanilla_pcnet"]:
        experiment = ExperimentPCNet(config)
        experiment.run()
        
    elif config["algorithm"] in ["attentional_pcnet_random_ensemble"]:
        experiment = ExperimentAttentionalPCNetRandomEnsemble(config)
        experiment.run()
        
    elif config["algorithm"] in ["random_ensemble_pcnet"]:
        experiment = ExperimentRandomEnsemblePCNet(config)
        experiment.run()
        
    elif config["algorithm"] in ["ge_pcnet", "ge_attentional_pcnet"]:
        experiment = ExperimentGEPC(config)
        experiment.run()
    
        
    elif config["algorithm"] in ["hybrid_pcnet"]:
        experiment = ExperimentHybridPCNet(config)
        experiment.run()

    elif config["algorithm"] in ["hybrid_pcnet_random_ensemble"]:
        experiment = ExperimentHybridPCNetRandomEnsemble(config)
        experiment.run()

    elif config["algorithm"] in ["attentional_robust_structural_pcnet"]:
        experiment = ExperimentAttentionalRobustStructuralPCNet(config)
        experiment.run()

    elif config["algorithm"] in ["robust_structural_pcnet"]:
        experiment = ExperimentRobustStructuralPCNet(config)
        experiment.run()
    
    elif config["algorithm"] in ["multihead_attentional_pcnet"]:
        experiment = ExperimentMultiHeadAttentionalPCNet(config)
        experiment.run()
    else:
        raise ValueError(f"Algorithm {config['algorithm']} not supported")
    
# ============================================================
# Loss Functions
# ============================================================

class DiceLoss(nn.Module):
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
        return  0.6* self.ce(pred, target) + 0.4*self.dice_loss(pred, target)

class LogDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def log_dice_loss(self, pred, target):
        C = pred.size(1)
        if C == 1:
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()
        else:
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (pred_prob * target).sum(dim=(2, 3))
        denominator = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        loss = -torch.log(torch.clamp(dice, min=1e-6))
        return loss.mean()

    def focal_loss(self, pred, target):
        """
        Generalized Focal Loss using torch.nn.functional.cross_entropy only.
        Works for both binary (C=1) and multi-class (C>1) segmentation.

        Args:
            pred:   [B, C, H, W] logits
            target: [B, H, W] or [B, 1, H, W] integer class labels
        """
        # If target has channel dimension, squeeze it
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        target = target.long()
        # Handle binary as 2-class cross-entropy (C=1 → expand to C=2)
        if pred.size(1) == 1:
            # Convert single-logit binary case to two-class logits for CE
            pred = torch.cat([pred, -pred], dim=1)

        # Compute per-pixel cross entropy (no reduction)
        ce_loss = F.cross_entropy(pred, target.long(), reduction="none")  # [B, H, W]

        # Compute pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        # Alpha handling
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        elif isinstance(self.alpha, torch.Tensor) and self.alpha.numel() == pred.size(1):
            alpha_t = self.alpha[target]  # [B, H, W]
        else:
            raise ValueError("alpha must be scalar or a per-class tensor with length = n_classes")

        # Focal loss
        focal = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


    def forward(self, pred, target):
        log_dice = self.log_dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return 0.5 * log_dice + 0.5 * focal
    
import torch
import torch.nn.functional as F

def soft_erode(x):
    # 3x3 min-pool approximation via -maxpool(-x)
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

def soft_skeletonize(p, iters=3):
    # p in [0,1]; iterative soft skeletonization
    skel = torch.zeros_like(p)
    img = p
    for _ in range(iters):
        er = soft_erode(img)
        op = F.relu(img - er)
        skel = torch.maximum(skel, op)
        img = er
    return skel

def soft_clDice(pred_prob, target, iters=3, eps=1e-6):
    # pred_prob, target: (B,1,H,W) in [0,1]
    S_p = soft_skeletonize(pred_prob, iters)
    S_t = soft_skeletonize(target, iters)
    tprec = (S_p * target).sum(dim=(1,2,3)) / (S_p.sum(dim=(1,2,3)) + eps)
    tsens = (S_t * pred_prob).sum(dim=(1,2,3)) / (S_t.sum(dim=(1,2,3)) + eps)
    cldice = (2 * tprec * tsens) / (tprec + tsens + eps)
    return cldice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=pos_weight)  # or BCEWithLogitsLoss for binary
        self.gamma = gamma

    def dice_loss(self, pred, target, smooth=1.0):
        if pred.size(1) == 1:
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()
        else:
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=pred.size(1)).permute(0,3,1,2).float()
        inter = (pred_prob * target).sum(dim=(2,3))
        denom = pred_prob.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * inter + smooth) / (denom + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        if pred.size(1) == 1:
            if target.ndim == 3:
                target = target.unsqueeze(1)
            ce = F.binary_cross_entropy_with_logits(pred, target.float())
            pred_prob = torch.sigmoid(pred)
            tgt_prob = target.float()
            cl = soft_clDice(pred_prob, tgt_prob)
        else:
            ce = self.ce(pred, target)
            # treat class 1 as foreground for clDice; adapt if needed
            pred_prob = torch.softmax(pred, dim=1)[:,1:2]
            tgt_prob = (target==1).unsqueeze(1).float()
            cl = soft_clDice(pred_prob, tgt_prob)
        dl = self.dice_loss(pred, target)
        return 0.2*ce + 0.1*dl + self.gamma*(1 - cl)

class UnsupervisedPCNetLoss(nn.Module):
    """
    A stable, balanced, and segmentation-oriented unsupervised loss for crack detection.

    Key features:
      • symmetric background/crack distributions
      • gamma entropy term prevents collapse
      • sparsity term anchors crack ratio to realistic level
      • deviation-based crack modeling
      • logit temperature softens responsibilities
      • works with 1-channel or 2-channel PCNet outputs
    """

    def __init__(self,
                 target_crack_ratio=0.06,   # expected 3% pixels are crack (tune)
                 lambda_sparsity=0.2,       # weight for crack area prior
                 lambda_entropy=0.15,       # weight for gamma entropy
                 eps=1e-6,
                 temperature=2.0):          # softens gamma to avoid saturation
        super().__init__()
        self.target_r = target_crack_ratio
        self.lambda_s = lambda_sparsity
        self.lambda_e = lambda_entropy
        self.eps = eps
        self.T = temperature

    def forward(self, logits, x):
        """
        logits: (B,1,H,W) or (B,2,H,W)
        x:      (B,C,H,W)
        """

        # ---------------------------------------------------------
        # 1. RESPONSIBILITY γ
        # ---------------------------------------------------------
        if logits.shape[1] == 1:
            crack_logits = logits[:, 0]
            background_logits = torch.zeros_like(crack_logits)
        else:
            background_logits = logits[:, 0]
            crack_logits      = logits[:, 1]

        # Temperature-scaled logit difference to avoid saturation
        delta = (crack_logits - background_logits) / self.T
        delta = torch.clamp(delta, -8, 8)  # prevents extreme γ
        gamma = torch.sigmoid(delta)       # (B,H,W)

        gamma_full = gamma.unsqueeze(1).expand(-1, x.size(1), -1, -1)

        # ---------------------------------------------------------
        # 2. BACKGROUND MODEL (Gaussian-like)
        # ---------------------------------------------------------
        mu_b  = torch.mean(x, dim=[2,3], keepdim=True)
        std_b = torch.abs(x - mu_b) + self.eps   # local variance (balanced!)

        log_pb = -0.5 * (((x - mu_b) / std_b)**2 +
                         torch.log(2 * torch.pi * std_b**2))

        # ---------------------------------------------------------
        # 3. CRACK MODEL (deviation-based)
        # ---------------------------------------------------------
        # cracks are deviations from the background mean
        deviation = torch.abs(x - mu_b)

        # cracks get a moderate variance model centered around deviations
        std_c = deviation + self.eps
        mu_c  = mu_b

        log_pc = -0.5 * (((x - mu_c) / std_c)**2 +
                         torch.log(2 * torch.pi * std_c**2))

        # ---------------------------------------------------------
        # 4. MIXTURE PROBABILITY
        # ---------------------------------------------------------
        nll = -(gamma_full * log_pc + (1 - gamma_full) * log_pb)  # (B,C,H,W)
        nll = nll.mean(dim=1)                                     # (B,H,W)
        base_loss = nll.mean()

        # ---------------------------------------------------------
        # 5. SPARSITY PRIOR: penalize deviation from target crack ratio
        # ---------------------------------------------------------
        crack_ratio = gamma.mean()
        sparsity_prior = (crack_ratio - self.target_r).pow(2)

        # ---------------------------------------------------------
        # 6. ENTROPY REGULARIZATION on γ: keeps responsibilities soft
        # ---------------------------------------------------------
        entropy = -(gamma * torch.log(gamma + self.eps) +
                    (1 - gamma) * torch.log(1 - gamma + self.eps))
        entropy = entropy.mean()

        # ---------------------------------------------------------
        # FINAL LOSS
        # ---------------------------------------------------------
        loss = (
            base_loss +
            self.lambda_s * sparsity_prior -
            self.lambda_e * entropy          # **maximize entropy**
        )

        return loss

class Experiment:
    def __init__(self, config):
        self.config = config
        self.execution_mode = self.config.get("execution_mode", "train")
        self.device = self.config.get("device", "cpu")

        self.data_path = self.config.get("data_path", "data/train")
        self.validation_split = self.config.get("validation_split", 0.2)
        self.test_path = self.config.get("test_path", "data/test")
        self.save_path = self.config.get("save_path", "models/")

        self.batch_size = self.config.get("batch_size", 16)
        self.sample_size = self.config.get("sample_size", None)
        self.test_size = self.config.get("test_size", None)
        self.resize_size = self.config.get("resize_size", [256, 256])
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.threshold = self.config.get("threshold", 0.5)
        self.n_classes = self.config.get("n_classes", 2)
        self.preprocessing = self.config.get("preprocessing", False)
        
        self.epochs = self.config.get("epochs", 10)
        self.model = None
        self.lr = self.config.get("learning_rate", 0.001)

        self.log_path = self.config.get("log_dir", "logs/")
        print_configs("Experiment Configurations:", self.config)
        self.experiment_dict = {}
        self.experiment_report = os.path.join(self.log_path, "experiment_report.txt")
        with open(self.experiment_report, "w") as f:
            f.write(f"Experiment Configurations: {self.config}\n")
        f.close()
    

    
    @abc.abstractmethod
    def set_algorithm(self):
        pass

    def set_dataset(self,train_path, test_path=None):

        if self.execution_mode == "train":
            self.images_path = train_path + "/images/"
            self.ground_truth_path = train_path + "/masks/"  
            if self.sample_size is not None: 
                n_images = len(os.listdir(self.images_path))
                print_info(f"Number of images: {n_images}")
                self.sample_size = int(n_images * self.sample_size)
                print_info(f"Using sample size: {self.sample_size}/{n_images} which is {self.sample_size/n_images*100}% of the dataset")  
            self.data = CrackDataset(self.images_path, self.ground_truth_path,sample_size=self.sample_size, resize_size=(self.resize_size[0], self.resize_size[1]), preprocessing=self.preprocessing)
            torch.manual_seed(self.config.get("seed", 42))
            perm = torch.randperm(len(self.data))  # permutation over CrackDataset

            if self.execution_mode == "train":
                if self.validation_split > 0:
                    split_idx = int((1 - self.validation_split) * len(self.data))
                    train_idx = perm[:split_idx]
                    val_idx = perm[split_idx:]
                    self.train_data = torch.utils.data.Subset(self.data, train_idx)
                    self.val_data = torch.utils.data.Subset(self.data, val_idx)
                    print_info(f"Training samples: {len(self.train_data)}, Validation samples: {len(self.val_data)}")
                else:
                    self.train_data = torch.utils.data.Subset(self.data, perm)
                    self.val_data = None
                    print_info(f"Training samples: {len(self.train_data)}")
            if test_path is not None:
                self.test_images_path = test_path + "/images/"
                self.test_ground_truth_path = test_path + "/masks/"
                self.test_data = CrackDataset(self.test_images_path, self.test_ground_truth_path,sample_size=self.test_size, resize_size=(self.resize_size[0], self.resize_size[1]), preprocessing=self.preprocessing)
                print_info(f"Test samples: {len(self.test_data)}")
            self.experiment_dict["dataset_size"] = {
                "train": len(self.data),
                "val": len(self.val_data) if self.validation_split > 0 else 0,
                "test": len(self.test_data) if test_path is not None else 0
            }
        elif self.execution_mode == "test":
            self.test_images_path = test_path + "/images/"
            self.test_ground_truth_path = test_path + "/masks/"
            self.test_data = CrackDataset(self.test_images_path, self.test_ground_truth_path,sample_size=self.test_size, resize_size=(self.resize_size[0], self.resize_size[1]), preprocessing=self.preprocessing)
            print_info(f"Test samples: {len(self.test_data)}")
            self.experiment_dict["dataset_size"] = {
                "test": len(self.test_data)
            }

        else:
            raise ValueError(f"Execution mode {self.execution_mode} not supported")
        
        with open(self.experiment_report, "a") as f:
            f.write(f"Dataset sizes: {self.experiment_dict['dataset_size']}\n")
        f.close()

    # ----------------------------
    # Metrics
    # ----------------------------

    def dice_coef(self, logits, targets, smooth=1.0):
        """
        logits:  (B, C, H_pred, W_pred)
        targets: (B, H_gt,  W_gt) or (H, W)
        """

        # ---------------------------------------------------------
        # 1. Make sure targets has batch dimension
        # ---------------------------------------------------------
        if targets.dim() == 2:  # (H, W)
            targets = targets.unsqueeze(0)  # (1, H, W)

        # ---------------------------------------------------------
        # 2. Resize logits to match GT spatial size
        # ---------------------------------------------------------
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        B, C, H, W = logits.shape

        # ---------------------------------------------------------
        # 3. Binary Dice
        # ---------------------------------------------------------
        if C == 1:
            probs = torch.sigmoid(logits)              # (B,1,H,W)
            targets = targets.unsqueeze(1).float()     # (B,1,H,W)

            intersection = (probs * targets).sum(dim=(2,3))
            denominator   = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))

            dice = (2 * intersection + smooth) / (denominator + smooth)
            return dice.mean()

        # ---------------------------------------------------------
        # 4. Multi-class Dice (macro Dice)
        #    One-hot encode ground truth
        # ---------------------------------------------------------
        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)

        # one-hot: (B,H,W,C) → (B,C,H,W)
        targets_1h = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_1h).sum(dim=(2,3))      # (B,C)
        denominator  = probs.sum(dim=(2,3)) + targets_1h.sum(dim=(2,3))

        dice = (2 * intersection + smooth) / (denominator + smooth)  # (B,C)

        # Mean across classes + batch
        return dice.mean()

    def iou_score(self, logits, targets, num_classes=None):
        """
        logits:  (N, C, H, W) for multi-class OR (N, H, W) binary mask
        targets: (N, H, W) or (N, 1, H, W)
        """
        # 1. Squeeze channel dim from targets if present
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        # 2. Resize logits if spatial dimensions differ
        if logits.shape[-2:] != targets.shape[-2:]:
            was_3d = (logits.dim() == 3)
            if was_3d: logits = logits.unsqueeze(1)
            logits = torch.nn.functional.interpolate(
                logits.float(), 
                size=targets.shape[-2:], 
                mode="nearest" # Nearest is safer for masks/classes
            )
            if was_3d: logits = logits.squeeze(1)

        # 3. Convert to predicted class indices safely
        if logits.dim() == 4 and logits.shape[1] > 1:
            # Multi-class scenario: Take argmax across channels
            preds = torch.argmax(logits, dim=1)
            if num_classes is None: num_classes = logits.shape[1]
        else:
            # Binary mask scenario (already thresholded or single channel)
            preds = (logits > 0.5)
            if num_classes is None: num_classes = 2

        # 4. CRITICAL FIX: Flatten and cast to Long (Integer) for bincount
        preds = preds.contiguous().view(-1).long()
        targets = targets.contiguous().view(-1).long()

        # 5. Compute confusion matrix
        confusion = torch.bincount(
            num_classes * targets + preds,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        TP = confusion.diag()
        FP = confusion.sum(0) - TP
        FN = confusion.sum(1) - TP

        # Compute IoU per class
        IoU = TP / (TP + FP + FN + 1e-7)
        
        # For binary crack detection, we usually only care about the Foreground IoU (class 1)
        # But to keep it generic across your framework, we return the mean.
        return IoU.mean()

    def cl_iou_score(self, pred_mask, true_mask, tau=4):
        """
        Calculates the true Centerline-IoU (clIoU) using topological skeletonization.
        """
        import skimage.morphology as morph  # <-- Added missing import here
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
        
        # Squeeze channel dim if needed to ensure shape is (B, 1, H, W)
        if pred_mask.dim() == 3: pred_mask = pred_mask.unsqueeze(1)
        if true_mask.dim() == 3: true_mask = true_mask.unsqueeze(1)

        B = pred_mask.shape[0]
        device = pred_mask.device
        
        # 1. Binarize and move to CPU numpy for topological thinning
        pm_np = (pred_mask > 0.5).cpu().numpy().astype(bool)
        tm_np = (true_mask > 0.5).cpu().numpy().astype(bool)
        
        skel_pred_np = np.zeros_like(pm_np)
        skel_true_np = np.zeros_like(tm_np)
        
        # 2. Extract TRUE Centerlines (Iterative Thinning)
        for b in range(B):
            # skimage skeletonize expects a 2D boolean array
            skel_pred_np[b, 0] = morph.skeletonize(pm_np[b, 0])
            skel_true_np[b, 0] = morph.skeletonize(tm_np[b, 0])
            
        # Convert true centerlines back to PyTorch tensors
        skel_pred = torch.tensor(skel_pred_np, dtype=torch.float32, device=device)
        skel_true = torch.tensor(skel_true_np, dtype=torch.float32, device=device)

        # 3. Dilation
        k_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tau+1, 2*tau+1))
        k = torch.tensor(k_np, dtype=torch.float32, device=device)[None, None, :, :]
        
        dil_true = (F.conv2d(skel_true, k, padding=tau, groups=1) > 0).float()
        dil_pred = (F.conv2d(skel_pred, k, padding=tau, groups=1) > 0).float()

        # 4. Intersection Metrics
        TPcl = (skel_true * dil_pred).sum(dim=(1, 2, 3))
        FPcl = (skel_pred * (1 - dil_true)).sum(dim=(1, 2, 3))
        FNcl = (skel_true * (1 - dil_pred)).sum(dim=(1, 2, 3))

        cl_iou = TPcl / (TPcl + FPcl + FNcl + 1e-8)
        return cl_iou.mean().item()


    from torch.utils.data import DataLoader, TensorDataset

    def get_dataloaders(self, data, batch_size=None, labels=None, train_sampler=None, shuffle=True, num_workers=0):
        """
        If train_sampler is provided, shuffle must be False (sampler handles shuffling).
        """
        if labels is not None:
            dataset = TensorDataset(data, labels)
        else:
            dataset = data

        if train_sampler is not None:
            return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=0)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    
    def compute_pos_weight(self, masks):
        """
        Computes pos_weight = num_neg / num_pos for BCEWithLogitsLoss
        from a segmentation dataset.
        """
        # Class balance
        flat = masks.view(-1)
        counts = torch.bincount(flat, minlength=self.n_classes).float()
        counts[counts == 0] = 1.0
        balance = (1.0 / counts)
        balance = balance / balance.sum()
        return balance

    @abc.abstractmethod
    def setup(self, pos_weight=1.0):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def plot_losses(self, train_losses, val_losses=None):
        plt.plot(train_losses, label="Train Loss")
        if val_losses:
            plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.show()

    def log_results(self):
        pass

class ExperimentDL(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, pos_weight=1.0):
        self.set_optimizer(torch.optim.Adam(self.model.parameters(), lr=self.learning_rate))
        self.set_scheduler(torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1))
        self.set_loss(DiceCELoss())

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_loss(self, loss):
        self.loss = loss
    
    
    def get_model_size_kb(self,model: torch.nn.Module):
        """Return model size in kilobytes (KB)."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size_bytes = param_size + buffer_size
        return total_size_bytes / 1024  # Convert to KB
    # ----------------------------
    # Training loop
    # ----------------------------

    def iou_score(self, logits, targets, num_classes=None):

        # Resize if sizes do not match
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # Decide number of classes
        if num_classes is None:
            num_classes = 2 if logits.shape[1] == 1 else logits.shape[1]

        # Predictions
        if logits.shape[1] == 1:   # binary
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
        else:                      # multiclass
            preds = torch.argmax(logits, dim=1)

        # Flatten + convert
        preds = preds.view(-1).long()
        targets = targets.view(-1).long()

        # Compute confusion matrix
        bins = num_classes * targets + preds
        confusion = torch.bincount(
            bins,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        TP = confusion.diag()
        FP = confusion.sum(0) - TP
        FN = confusion.sum(1) - TP

        IoU = TP / (TP + FP + FN + 1e-7)
        return IoU.mean()

    def train(self, model, train_data, val_data=None, 
            loss_function=None, optimizer=None, scheduler=None, 
            device="cuda", epochs=20, batch_size=8):

        # Default loss: BCE + Dice
        if loss_function is None:
            loss_function = DiceCELoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader = self.get_dataloaders(train_data, batch_size=batch_size)
        val_loader = self.get_dataloaders(val_data, batch_size=batch_size) if val_data else None

        model.to(device)
        train_losses, val_losses, train_dices, train_clious, val_dices, train_ious, val_ious, val_clious = [], [], [], [], [], [], [] , []
        
        for epoch in trange(epochs):
            model.train()
            epoch_loss, epoch_dice, epoch_iou, epoch_cl_iou = 0, 0, 0, 0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks
                optimizer.zero_grad()
                outputs = model(images)  
                loss = loss_function(outputs, masks) 
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_dice += self.dice_coef(outputs, masks).item()
                epoch_iou  += self.iou_score(outputs, masks).item()
                epoch_cl_iou += self.cl_iou_score(outputs, masks)

            train_loss = epoch_loss / len(train_loader)
            train_dice = epoch_dice / len(train_loader)
            train_iou  = epoch_iou / len(train_loader)
            train_cl_iou = epoch_cl_iou / len(train_loader)
            train_losses.append(train_loss)
            train_dices.append(train_dice)
            train_ious.append(train_iou)
            train_clious.append(train_cl_iou)
            # Validation
            if val_loader:
                model.eval()
                val_loss, val_dice, val_iou, val_cl_iou = 0, 0, 0, 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        val_loss += loss_function(outputs, masks).item()
                        val_dice += self.dice_coef(outputs, masks).item()
                        val_iou  += self.iou_score(outputs, masks).item()
                        val_cl_iou += self.cl_iou_score(outputs, masks)

                val_loss /= len(val_loader)
                val_dice /= len(val_loader)
                val_iou  /= len(val_loader)
                val_cl_iou /= len(val_loader)

            else:
                val_loss, val_dice, val_iou = None, None, None
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_ious.append(val_iou)
            val_clious.append(val_cl_iou)
            if scheduler:
                scheduler.step()

            # Logging
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, CL IoU: {train_cl_iou:.4f} | "
                f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}, CL IoU: {val_cl_iou:.4f}" if val_loader else "")
            
        train_df = pd.DataFrame({"train_loss": train_losses, "train_dice": train_dices, "train_iou": train_ious, "train_cl_iou": train_clious})
        train_df.to_csv(os.path.join(self.log_path, "training.csv"), index=False)
        if val_loader:
            val_df = pd.DataFrame({"val_loss": val_losses, "val_dice": val_dices, "val_iou": val_ious, "val_cl_iou": val_clious})
            val_df.to_csv(os.path.join(self.log_path, "validation.csv"), index=False)
        
        
        with open(self.experiment_report, "a") as f:
            f.write(f"Final Training Loss: {train_losses[-1]}, Dice: {train_dices[-1]}, IoU: {train_ious[-1]}, CL IoU: {train_clious[-1]}\n")
            if val_loader:
                f.write(f"Final Validation Loss: {val_losses[-1]}, Dice: {val_dices[-1]}, IoU: {val_ious[-1]}, CL IoU: {val_clious[-1]}\n")
        f.close()
        return model

    def finetune(self, model, dataloader, n_classes=2):
        model.load_state_dict(torch.load("unet_pretrained.pth"))

        # Replace the last layer for new classes
        model.conv_last = nn.Conv2d(64, n_classes, 1)

        # Option 1: freeze earlier layers (useful if the dataset is small)
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the last convolution block(s)
        for param in model.dconv_up1.parameters():
            param.requires_grad = True
        for param in model.conv_last.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        criterion = self.loss

        for images, masks in dataloader:
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def validate(self, model, val_loader, loss_function):
        model.eval()
        epoch_val_loss = 0  
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
                epoch_val_loss += loss.item()
        model.train()
        return epoch_val_loss

    def evaluate(self, model, test_data, loss_function=torch.nn.BCEWithLogitsLoss()):
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        model.eval()
        losses, dices, ious, clious = [], [], [], []
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        losses, dices, ious, clious = [], [], [], []
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
        
                dice_score = self.dice_coef(outputs, masks)
                iou_score = self.iou_score(outputs, masks)
                cl_iou_score = self.cl_iou_score(outputs, masks)
                prob = torch.sigmoid(outputs)
                pred_mask = (prob > self.threshold).float()
                true_mask = masks.float()

                # Flatten predictions and targets
                pm = pred_mask.view(-1).long()
                tm = true_mask.view(-1).long()

                tp = ((pm == 1) & (tm == 1)).sum()
                fp = ((pm == 1) & (tm == 0)).sum()
                fn = ((pm == 0) & (tm == 1)).sum()
                tn = ((pm == 0) & (tm == 0)).sum()

                # Accuracy
                accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

                # Precision = TP / (TP + FP)
                precision = tp / (tp + fp + 1e-8)

                # Recall = TP / (TP + FN)
                recall = tp / (tp + fn + 1e-8)

                # F1 = harmonic mean
                f1_score = 2 * precision * recall / (precision + recall + 1e-8)

                losses.append(loss.item())
                dices.append(float(torch.as_tensor(dice_score).mean()))
                ious.append(float(torch.as_tensor(iou_score).mean()))
                clious.append(float(torch.as_tensor(cl_iou_score)))

                accuracies.append(accuracy.item())
                precisions.append(precision.item())
                recalls.append(recall.item())
                f1_scores.append(f1_score.item())

        print_info("✅ Testing complete.")
        print_info(f"Test Loss: {np.mean(losses):.4f}, Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}, CL IoU: {np.mean(clious):.4f}")
        print_info(f"Test Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1-Score: {np.mean(f1_scores):.4f}")
        self.experiment_dict["results"] = {
            "test_loss": np.mean(losses),
            "test_dice": np.mean(dices),
            "test_iou": np.mean(ious),
            "test_cl_iou": np.mean(clious),
            "test_accuracy": np.mean(accuracies),
            "test_precision": np.mean(precisions),
            "test_recall": np.mean(recalls),
            "test_f1_score": np.mean(f1_scores)
        }
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.experiment_dict["n_trainable_params"] = n_trainable_params
        with open(self.experiment_report, "a") as f:
            f.write(f"Test Loss: {np.mean(losses):.4f}, Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}, CL IoU: {np.mean(clious):.4f}\n")
            f.write(f"Test Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1-Score: {np.mean(f1_scores):.4f}\n")
        test_loss = sum(losses)

                
        return test_loss/len(test_loader)        

    def log_results(self):
        pass

    def show_predictions(self, model, dataset, device, num_samples=3, threshold=0.5):
        # Save model info
        model_json = os.path.join(self.log_path, "model_info.json")
        model_memory = self.get_model_size_kb(model)
        with open(self.experiment_report, "a") as f:
            f.write(f"Model info: {model_json}\n")
            f.write(f"Model size: {model_memory} KB\n")
        f.close()

        
        predictions_dir = os.path.join(self.log_path, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # We’ll create up to num_samples * 3 total images per grid
        total_samples = len(dataset)
        num_groups = total_samples // num_samples
        max_pred = 15
        for j in range(num_groups):
            if j > max_pred:
                print(f"Saved {max_pred} prediction grids.")
                break
            fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 9))  # 3 rows × num_samples columns

            for i in range(num_samples):
                idx = j * num_samples + i
                if idx >= total_samples:
                    # Hide unused axes if dataset doesn't divide evenly
                    for row in range(3):
                        axes[row, i].axis("off")
                    continue

                img, gt = dataset[idx]
                img = img.unsqueeze(0).to(device)
                gt_np = gt.squeeze().cpu().numpy()

                # Prediction
                with torch.no_grad():
                    pred = model(img).cpu().squeeze()
                    prob = torch.sigmoid(pred).numpy()
                pred_mask = (prob > threshold).astype("uint8")

                # Convert input image for plotting
                img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

                # Row 1: Input
                axes[0, i].imshow(img_np)
                axes[0, i].set_title(f"Input {idx}")
                axes[0, i].axis("off")

                # Row 2: Ground Truth
                axes[1, i].imshow(gt_np, cmap="gray")
                axes[1, i].set_title(f"Ground Truth {idx}")
                axes[1, i].axis("off")

                # Row 3: Prediction
                axes[2, i].imshow(pred_mask, cmap="cool")
                axes[2, i].set_title(f"Prediction {idx}")
                axes[2, i].axis("off")
            plt.savefig(os.path.join(predictions_dir, f"predictions_grid_{j}.png"), bbox_inches="tight")
            plt.close(fig)
            

    @abc.abstractmethod
    def save_model(self, model_name="model"):
        pass

    @abc.abstractmethod
    def load_model(self, model_name="model"):
        pass

    def run(self):
        self.set_algorithm()
        self.model.to(self.device)
        
        
        if self.execution_mode == "train":
            self.set_dataset(self.data_path, self.test_path)
            # get the class imbalance ratio
            images = [img for img, mask in self.train_data]
            masks = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
            masks = torch.from_numpy(masks).long().to(self.device)
            pos_weight = self.compute_pos_weight(masks)
            pos_weight = pos_weight[1]/pos_weight[0]
            self.setup(pos_weight=pos_weight)
            self.model = self.train(self.model, train_data=self.train_data, val_data=self.val_data, loss_function=self.loss, optimizer=self.optimizer, device=self.device, epochs=self.epochs, batch_size=self.batch_size)
            self.save_model(self.model_name)
            if self.test_path is not None:
                self.evaluate(self.model, self.test_data, loss_function=self.loss)
                json.dump(self.experiment_dict, open(os.path.join(self.log_path, "experiment_results.json"), "w"), indent=4)
                self.show_predictions(self.model, self.test_data, self.device, num_samples=2, threshold=self.threshold)
        elif self.execution_mode == "finetune":
            self.set_dataset(self.data_path, self.test_path)
            pos_weight = self.compute_pos_weight(self.data)
            self.set_algorithm()
            self.load_model()
            self.setup(pos_weight=pos_weight)
            self.model.to(self.device)
            self.finetune()

        elif self.execution_mode == "test":
            self.set_dataset(self.data_path, self.test_path)
            pos_weight = self.compute_pos_weight(self.data)
            self.set_algorithm()
            self.load_model(self.model_name)
            self.setup(pos_weight=pos_weight)
            self.model.to(self.device)
            print(self.model.info())
            self.evaluate(self.model, self.test_data, loss_function=self.loss)
            json.dump(self.experiment_dict, open(os.path.join(self.log_path, "experiment_results.json"), "w"), indent=4)
            self.show_predictions(self.model, self.test_data, self.device, num_samples=2, threshold=self.threshold)

class UNetExperiment(ExperimentDL):

    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unet")

    def set_algorithm(self):
        self.model = UNet(n_classes=1)
    
    def save_model(self, model_name="unet"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="unet"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ConvAEExperiment(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.input_channels = self.config.get("input_channels", 3)
        self.hidden_channels = self.config.get("hidden_channels", (128, 64))
        self.n_classes = self.config.get("n_classes", 1)
        self.kernel_size = self.config.get("kernel_size", 3)
        self.stride = self.config.get("stride", 1)
        self.padding = self.config.get("padding", 9)
        self.dropout = self.config.get("dropout", 0.2)
        self.model_name = self.config.get("model_name", "convae")
        self.filter_size = self.config.get("filter_size", (3,7))

    
    def set_algorithm(self):
        self.model = ConvAutoEncoder(input_channels=self.input_channels, hidden_channels=self.hidden_channels, n_classes=self.n_classes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dropout=self.dropout)

    def save_model(self, model_name="convae"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="convae"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentUnetPlus(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unetplus")
        

    def set_algorithm(self):
        self.model = UNetPlusPlus(n_classes=1)

    def save_model(self, model_name="unetplus"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="unetplus"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentCrackFormer(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "crackformer")
    
    def set_algorithm(self):
        self.model = crackformer()

    def save_model(self, model_name="crackformer"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="crackformer"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentDeepCrackZ(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "deepcrackL")
    
    def set_algorithm(self):
        self.model = DeepCrack()

    def save_model(self, model_name="deepcrackL"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="deepcrackL"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentPC(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "pcnet")
        
        # PCNet Specific Config
        self.n_classes = self.config.get("n_classes", 2)
        self.max_depth = self.config.get("max_depth", 4)
        self.max_branching = self.config.get("max_branching", 2)
        
        # Experiment Config
        self.epochs = self.config.get("epochs", 10)
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 1e-2)
        self.seed = self.config.get("seed", 42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_type = self.config.get("data_type", "image")  # 'image' or 'tabular'
        
        # State flags
        self.is_initialized = False
        
        # Keyword arguments for the model's forward pass
        self.forward_kwargs = {}
        
        # Initialize the algorithm wrapper (PCNet itself is built in train/init_network)
        self.set_algorithm()

    def set_algorithm(self):
        pass
    # --------------------------------------------------------
    # Data Handling
    # --------------------------------------------------------
    def _prepare_input(self, x):
        """
        Unifies input shapes for PCNet:
        - Image: (B, C, H, W) -> Stays (B, C, H, W)
        - Tabular: (B, D)     -> Becomes (B, D, 1, 1)
        """
        if x.dim() == 2:
            return x.unsqueeze(-1).unsqueeze(-1)
        return x

    # --------------------------------------------------------
    # Pipeline Orchestration (The requested method)
    # --------------------------------------------------------
    def run(self):
        """
        Main entry point for the experiment pipeline.
        Handles Setup -> Data -> Train/Test -> Save/Viz.
        """
        # 1. Environment & Seeding
        if hasattr(self, '_setup_cpu_env'):
            self._setup_cpu_env()
        else:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        print_info(f"🚀 Execution Mode: {self.execution_mode.upper()} | Device: {self.device}")

        # 2. Data Loading
        # self.set_dataset must populate self.train_loader / self.test_loader
        self.set_dataset(self.data_path, self.test_path)
        
        # 3. Execution Flow
        if self.execution_mode == "train":
            # A. Training Phase
            # Note: init_network() happens inside train() on the first batch
            self.train()
            
            # B. Save the learned structure and weights
            self.save_model()
            if self.val_data is not None:
                val_loader = self.get_dataloaders(self.val_data, batch_size=self.batch_size)
                self.calibrate_threshold(val_loader)
            print_info("\n📊 Running Post-Training Evaluation...")
            self.test_loader = self.get_dataloaders(self.test_data, batch_size=self.batch_size)
            self.test(self.model, self.test_loader)
            self.show_prediction(self.model)

        elif self.execution_mode == "test":
            # A. Load Model 
            # Crucial: Must load the full object to restore the dynamic graph structure
            if not os.path.exists(os.path.join(self.log_path, f"{self.model_name}.pkl")):
                print_info(f"❌ Model not found at {self.log_path}. Please train first.")
                return
            
            self.load_model()
            self.test_loader = self.get_dataloaders(self.test_data, batch_size=self.batch_size)
            self.calibrate_threshold(self.test_loader)
            # B. Evaluate
            self.test(self.model, self.test_loader)
            
            # C. Visualize
            self.show_prediction(self.model)
        
        else:
            raise ValueError(f"Unknown execution_mode: {self.execution_mode}")

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    def train(self):
        print_info(f"⚙️ Starting Training for {self.epochs} epochs...")
        
        # FIX: Do NOT create the optimizer here. 
        # The model has no parameters yet! They are created in init_network().
        optimizer = None
        
        self.model.train()
        self.train_loader = self.get_dataloaders(self.train_data, batch_size=self.batch_size)
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for i, batch in enumerate(self.train_loader):
                # Handle (x, y) or just x
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                x = self._prepare_input(x)

                # --- Dynamic Initialization ---
                # PCNet needs the first batch of data to build its graph structure
                if not self.is_initialized:
                    print_info(f"🔨 Initializing PCNet structure with input shape {x.shape}...")
                    self.model.init_network(x)
                    self.model = self.model.to(self.device)
                    
                    # FIX: Create the optimizer NOW, after parameters exist
                    optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    self.is_initialized = True

                # Safety check: If for some reason init failed, skip
                if optimizer is None:
                    continue

                optimizer.zero_grad()
                
                # --- Forward & Loss ---
                # PCNet outputs Log-Likelihoods
                log_probs = self.model(x, **self.forward_kwargs)
                
                # Maximizing Likelihood == Minimizing Negative Log-Likelihood
                loss = -torch.mean(log_probs)
                
                loss.backward()
                
                # Gradient clipping to stabilize learning of unconstrained log_sigmas
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1

            # Avoid division by zero if loader is empty
            avg_loss = epoch_loss / max(1, batch_count)
            print_info(f"Epoch [{epoch+1}/{self.epochs}] | NLL Loss: {avg_loss:.4f}")

        # Save the trained model
        self.save_model()
    # --------------------------------------------------------
    # Testing & Metrics
    # --------------------------------------------------------
    def dice_coef(self, preds, targets, smooth=1e-6):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
    def calibrate_threshold(self, val_loader):
        print_info("🔍 Calibrating optimal threshold to maximize F1-Score...")
        self.model.eval()
        all_probs, all_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1]
                else: continue
                
                log_probs = self.model(self._prepare_input(x), **self.forward_kwargs)
                anomaly_map = -log_probs
                
                # Squeeze extra dims
                if anomaly_map.dim() == 5 and anomaly_map.shape[2] == 1: 
                    anomaly_map = anomaly_map.squeeze(2)
                if anomaly_map.dim() == 4 and anomaly_map.shape[1] > 1: 
                    anomaly_map = anomaly_map.mean(dim=1, keepdim=True)
                if anomaly_map.shape[-2:] != y.shape[-2:]:
                    anomaly_map = F.interpolate(anomaly_map, size=y.shape[-2:], mode='bilinear', align_corners=False)
                
                # Min-Max Normalization per batch
                B = anomaly_map.shape[0]
                flat = anomaly_map.view(B, -1)
                mins = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
                maxs = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
                probs = (anomaly_map - mins) / (maxs - mins + 1e-8)
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y.numpy())
                
        # Flatten all predictions and targets across the entire validation set
        all_probs = np.concatenate(all_probs).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        
        # Sweep thresholds from 0.02 to 0.98
        best_f1, best_t = 0.0, 0.5
        thresholds = np.linspace(0.02, 0.98, 50)
        
        for t in thresholds:
            preds = (all_probs > t).astype(int)
            tp = ((preds == 1) & (all_targets == 1)).sum()
            fp = ((preds == 1) & (all_targets == 0)).sum()
            fn = ((preds == 0) & (all_targets == 1)).sum()
            
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                
        self.threshold = best_t  # Save the optimal threshold!
        print_info(f"✅ Calibration complete! Best Threshold: {self.threshold:.4f} (Val F1: {best_f1:.4f})")

    # =========================================================
    # Updated Test Method
    # =========================================================
    @torch.no_grad()
    def test(self, model, dataloader):
        model.eval()
        model.to(self.device)
        
        # Metric Lists
        losses, dices, ious, clious = [], [], [], []
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        
        print_info("🧪 Starting Evaluation with Full Metrics...")
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, masks = batch[0], batch[1]
            else:
                # If no masks provided, we can't compute these metrics
                continue

            images = images.to(self.device)
            masks = masks.to(self.device)

            # 1. Forward Pass (Get Log-Likelihood)
            # Output shape: (B, 1, H, W) or (B, 2, H, W)
            log_probs = model(self._prepare_input(images), **self.forward_kwargs)
            
            # 2. Calculate Loss (NLL)
            # We treat NLL as the "loss" for reporting
            loss = -torch.mean(log_probs)
            
            # 3. Process Outputs for Segmentation Metrics
            # A. Convert Log-Likelihood to Anomaly Score (Higher = More Anomalous)
            anomaly_map = -log_probs
            
            # Squeeze extra dims if log_probs is (B, C, 1, H, W)
            if anomaly_map.dim() == 5 and anomaly_map.shape[2] == 1:
                anomaly_map = anomaly_map.squeeze(2)
                
            if anomaly_map.dim() == 4 and anomaly_map.shape[1] > 1:
                anomaly_map = anomaly_map.mean(dim=1, keepdim=True)
                
            # Upsample the anomaly map to match the target mask spatial resolution if it differs
            if anomaly_map.shape[-2:] != masks.shape[-2:]:
                anomaly_map = F.interpolate(anomaly_map, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # B. Normalize Score to [0, 1] for Thresholding
            # (Min-Max normalization per batch to create a pseudo-probability)
            B = anomaly_map.shape[0]
            flat = anomaly_map.contiguous().view(B, -1)
            mins = flat.min(dim=1, keepdim=True)[0]
            maxs = flat.max(dim=1, keepdim=True)[0]
            
            # Dynamically reshape to match anomaly_map's dimensions for safe broadcasting
            view_shape = [B] + [1] * (anomaly_map.dim() - 1)
            mins = mins.view(*view_shape)
            maxs = maxs.view(*view_shape)
            
            probs = (anomaly_map - mins) / (maxs - mins + 1e-8)
            
            # C. Generate Predictions
            pred_mask = (probs > self.threshold).float()
            true_mask = masks.float()
            
            # Ensure shapes match (handle channel dims and spatial mismatches)
            if pred_mask.shape[-2:] != true_mask.shape[-2:]:
                pred_mask = F.interpolate(pred_mask, size=true_mask.shape[-2:], mode='nearest')
                
            if pred_mask.shape != true_mask.shape:
                # Usually masks are (B, H, W), preds are (B, 1, H, W)
                if pred_mask.dim() >= 2 and pred_mask.shape[1] == 1: pred_mask = pred_mask.squeeze(1)
                if true_mask.ndim == 4 and true_mask.shape[1] == 1: true_mask = true_mask.squeeze(1)

            # 4. Compute Metrics
            # Dice / IoU (using float masks)
            dice = self.dice_coef(pred_mask, true_mask)
            iou = self.iou_score(pred_mask, true_mask)
            cl_iou = self.cl_iou_score(pred_mask, true_mask)

            # Classification Stats (using flattened binary vectors)
            pm = pred_mask.contiguous().view(-1).long()
            tm = true_mask.contiguous().view(-1).long()

            tp = ((pm == 1) & (tm == 1)).sum().float()
            fp = ((pm == 1) & (tm == 0)).sum().float()
            fn = ((pm == 0) & (tm == 1)).sum().float()
            tn = ((pm == 0) & (tm == 0)).sum().float()

            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)

            # 5. Append Results
            losses.append(loss.item())
            dices.append(dice.item())
            ious.append(iou.item())
            clious.append(cl_iou)
            
            accuracies.append(accuracy.item())
            precisions.append(precision.item())
            recalls.append(recall.item())
            f1_scores.append(f1_score.item())

        # 6. Logging & Saving
        results = {
            "test_loss": np.mean(losses),
            "test_dice": np.mean(dices),
            "test_iou": np.mean(ious),
            "test_cl_iou": np.mean(clious),
            "test_accuracy": np.mean(accuracies),
            "test_precision": np.mean(precisions),
            "test_recall": np.mean(recalls),
            "test_f1_score": np.mean(f1_scores)
        }

        print_info("✅ Testing complete.")
        print_info(f"Test Loss: {results['test_loss']:.4f}, Dice: {results['test_dice']:.4f}, IoU: {results['test_iou']:.4f}, CL IoU: {results['test_cl_iou']:.4f}")
        print_info(f"Test Accuracy: {results['test_accuracy']:.4f}, Precision: {results['test_precision']:.4f}, Recall: {results['test_recall']:.4f}, F1-Score: {results['test_f1_score']:.4f}")

        self.experiment_dict["results"] = results
        
        # Calculate Params
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.experiment_dict["n_trainable_params"] = n_trainable_params
        
        # Write to report
        with open(self.experiment_report, "a") as f:
            f.write(f"Test Loss: {results['test_loss']:.4f}, Dice: {results['test_dice']:.4f}, IoU: {results['test_iou']:.4f}, CL IoU: {results['test_cl_iou']:.4f}\n")
            f.write(f"Test Accuracy: {results['test_accuracy']:.4f}, Precision: {results['test_precision']:.4f}, Recall: {results['test_recall']:.4f}, F1-Score: {results['test_f1_score']:.4f}\n")

        return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious)), float(np.mean(clious))

    # --------------------------------------------------------
    # I/O Helpers
    # --------------------------------------------------------
    def save_model(self):
        """Pickles the entire model object (structure + weights)"""
        save_path = os.path.join(self.log_path, f"{self.model_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
        print_info(f"💾 Model saved to {save_path}")

    def load_model(self):
        """Loads the pickled model object"""
        load_path = os.path.join(self.log_path, f"{self.model_name}.pkl")
        print_info(f"📂 Loading model from {load_path}")
        with open(load_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_initialized = True

    def show_prediction(self, model):
        """
        Visualizes: Input | Ground Truth | Anomaly Score (NLL) | Predicted Mask
        """
        if self.data_type == "tabular": 
            return

        print_info("🖼️ Generating prediction visualizations...")
        model.eval()
        save_dir = os.path.join(self.log_path, "predictions")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Get Data
        try:
            # We need the full batch (x, y) to show Ground Truth
            batch = next(iter(self.test_loader))
        except StopIteration:
            return

        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None # Unsupervised only (no GT)

        x = x.to(self.device)
        
        # 2. Model Inference
        with torch.no_grad():
            log_probs = model(self._prepare_input(x), **self.forward_kwargs)
            
            # Raw Anomaly Score (Negative Log Likelihood)
            anomaly_map = -log_probs 
            
            # Squeeze extra dims if log_probs is (B, C, 1, H, W)
            if anomaly_map.dim() == 5 and anomaly_map.shape[2] == 1:
                anomaly_map = anomaly_map.squeeze(2)
            
            # Collapse channels if necessary (e.g. if n_classes=2)
            if anomaly_map.dim() == 4:
                anomaly_map = anomaly_map.mean(dim=1) # (B, H, W)

        # 3. Calculate Threshold for Binary Classification
        # In a real scenario, this is calibrated on a validation set.
        # Here, we use the 95th percentile of the BATCH to highlight the "worst"
        # Create Binary Mask
        pred_mask = (anomaly_map > self.threshold).float()

        # 4. Plotting
        n_vis = min(4, x.shape[0])
        fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4*n_vis)) # 4 Columns
        
        # Handle single sample case (axes logic)
        if n_vis == 1: axes = np.expand_dims(axes, axis=0)

        for i in range(n_vis):
            # --- Prepare Data ---
            # A. Input Image
            img = x[i].cpu().permute(1, 2, 0).numpy()
            if img.shape[2] == 1: img = img.squeeze()
            
            # B. Ground Truth (if exists)
            if y is not None:
                gt = y[i].cpu().numpy()
                if gt.ndim == 3: gt = gt.squeeze()
            else:
                gt = np.zeros_like(img) # Placeholder

            # C. Anomaly Score (Heatmap)
            score = anomaly_map[i].cpu().numpy()
            
            # D. Predicted Mask (Binary)
            p_mask = pred_mask[i].cpu().numpy()

            # --- Plot Columns ---
            
            # Col 1: Input
            ax = axes[i, 0]
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title("Input Image")
            ax.axis('off')
            
            # Col 2: Ground Truth
            ax = axes[i, 1]
            ax.imshow(gt, cmap='gray')
            ax.set_title("Ground Truth")
            ax.axis('off')
            
            # Col 3: Anomaly Score (Heatmap)
            ax = axes[i, 2]
            im = ax.imshow(score, cmap='inferno') # 'inferno' highlights high values well
            ax.set_title("Anomaly Score (NLL)")
            ax.axis('off')
            # Optional: Add colorbar to the last plot or individually
            # plt.colorbar(im, ax=ax) 

            # Col 4: Predicted Mask
            ax = axes[i, 3]
            ax.imshow(p_mask, cmap='Reds', interpolation='nearest')
            ax.set_title(f"Pred Mask (>{self.threshold:.1f})")
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "vis_full_pipeline.png")
        plt.savefig(save_path)
        plt.close()
        print_info(f"✅ Full visualization saved to {save_path}")

class ExperimentGEPC(ExperimentPC):
    def __init__(self, config):
        config["model_name"] = config.get("model_name", "ge_pcnet")
        super().__init__(config)
        
    def set_algorithm(self):
        # We don't initialize a static PCNet here, the GE optimizer will build one.
        self.model = None

    def train(self):
        from probabilistic_circuits.grammatical_evolution import GEOptimizer
        print_info(f"⚙️ Starting GE Optimization for {self.epochs} generations...")
        
        ge = GEOptimizer(
            pop_size=self.config.get("pop_size", 16),
            genome_len=self.config.get("genome_len", 64),
            max_depth=self.config.get("max_depth", 4),
            max_branching=self.config.get("max_branching", 4),
            crossover_rate=self.config.get("crossover_rate", 0.9),
            mutation_rate=self.config.get("mutation_rate", 0.05),
            elite_k=self.config.get("elite_k", 2),
            n_classes=self.config.get("n_classes", 2),
            device=self.device,
            seed=self.config.get("seed", 42),
            n_jobs=self.config.get("n_cpus", os.cpu_count()),
            log_path=self.config.get("log_dir", "logs/ge_pcnet"),
            model_type="attentional_pcnet" if "attentional" in self.config.get("algorithm", "") else "pcnet"
        )

        # Pull all training data into memory for GE
        # Typically GE expects (N, C, H, W) and (N,)
        self.train_loader = self.get_dataloaders(self.train_data, batch_size=len(self.train_data))
        for batch in self.train_loader:
            x_train = batch[0].to(self.device)
            y_train = batch[1].to(self.device) if isinstance(batch, (list, tuple)) else None
            break
            
        x_train_prep = self._prepare_input(x_train)

        # Using the custom DiceCELoss with soft_clDice
        criterion = DiceCELoss()
            
        best_pc, best_genome, metrics = ge.evolve(
            inputs=x_train_prep,
            labels=y_train,
            criterion=criterion,
            generations=self.epochs,
            n_classes=self.config.get("n_classes", 2),
            train_steps=self.config.get("ge_train_steps", 10),
            lr=self.learning_rate
        )

        self.model = best_pc
        self.experiment_dict["ge_best_genome"] = best_genome
        self.experiment_dict["ge_metrics"] = metrics
        
        # Save the finalized GE model
        self.save_model()


class ExperimentPCNet(ExperimentPC):
    def __init__(self, config):
        config["model_name"] = config.get("model_name", "vanilla_pcnet")
        super().__init__(config)
        
    def set_algorithm(self):
        self.model = PCNet(
            n_classes=self.config.get("n_classes", 2),
            max_depth=self.config.get("max_depth", 4),
            max_branching=self.config.get("max_branching", 3),
            seed=self.config.get("seed", 42)
        )
        
class ExperimentAttentionalPCNet(ExperimentPC):
    def __init__(self, config):
        config["model_name"] = config.get("model_name", "attentional_pcnet")
        super().__init__(config)
        
    def set_algorithm(self):
        from probabilistic_circuits.probabilistic_circuits import AttentionalPCNet
        self.model = AttentionalPCNet(
            in_channels=self.config.get("input_channels", 3),
            n_classes=self.config.get("n_classes", 2),
            max_depth=self.config.get("max_depth", 4),
            max_branching=self.config.get("max_branching", 3),
            seed=self.config.get("seed", 42),
            embed_dim=self.config.get("embed_dim", 32)
        )

class ExperimentAttentionalPCNetRandomEnsemble(ExperimentPC):
    def __init__(self, config):
        config["model_name"] = config.get("model_name", "attentional_pcnet_random_ensemble")
        super().__init__(config)
        
    def set_algorithm(self):
        from probabilistic_circuits.probabilistic_circuits import RandomEnsembleAttentionalPCNet
        self.model = RandomEnsembleAttentionalPCNet(
            n_models=self.config.get("n_models", 5),
            device=self.device,
            input_channels=self.config.get("input_channels", 3),
            n_classes=self.config.get("n_classes", 2),
            max_depth=self.config.get("max_depth", 4),
            max_branching=self.config.get("max_branching", 3),
            seed=self.config.get("seed", 42),
            embed_dim=self.config.get("embed_dim", 32),
            ensemble_mode=self.config.get("ensemble_mode", "diversity"),
            diversity_weight=self.config.get("diversity_weight", 1.0)
        )

class ExperimentMultiHeadAttentionalPCNet(ExperimentPC):
    """
    Experiment running the Multi-Head Attentional PCNet.
    This version uses multiple 'heads' in each Sum node to attend to different 
    feature subspaces of the input image simultaneously.
    """
    def __init__(self, config):
        # Default name if not provided
        config["model_name"] = config.get("model_name", "multihead_attentional_pcnet")
        super().__init__(config)
        
        # Multi-Head specific hyperparameters
        self.embed_dim = self.config.get("embed_dim", 32)
        self.num_heads = self.config.get("num_heads", 4)
        
    def set_algorithm(self):
        """
        Instantiates the Multi-Head Attentional model.
        Note: The actual graph structure is built during the first forward pass 
        via init_network() in the parent train() method.
        """
        # Ensure the architecture class 'MultiHeadAttentionalPCNet' is available
        self.model = MultiHeadAttentionalPCNet(
            in_channels=self.config.get("input_channels", 3),
            n_classes=self.config.get("n_classes", 2),
            max_depth=self.config.get("max_depth", 4),
            max_branching=self.config.get("max_branching", 3),
            seed=self.config.get("seed", 42),
            embed_dim=self.config.get("embed_dim", 32),
            num_heads=self.config.get("num_heads", 4)
        )

    def log_results(self):
        """
        Optional: Custom logging to check head-weight distributions.
        This helps confirm if the heads are actually specializing or collapsing.
        """
        super().log_results()
        print_info("📊 Checking Multi-Head specialization...")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "head_mix" in name:
                    probs = torch.softmax(param, dim=0)
                    print_info(f"Node: {name} | Head Weights: {probs.cpu().numpy()}")