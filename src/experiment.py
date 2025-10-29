import mlflow.pytorch
import numpy as np
import cv2
import os
import mlflow
from networks import *
from probabilistic_circuits.probabilistic_circuits import *
from probabilistic_circuits.grammatical_evolution import *
import abc
from torch.utils.data import DataLoader
from tqdm import trange
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import *
import pickle
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from skimage.filters import frangi

from joblib import Parallel, delayed
import json

import pandas as pd
from crackdata_processing import CrackDataset, CrackAugmentations

from cv_modules.factories import CVModuleFactory

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch import multiprocessing as mp
import multiprocessing
import time

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

    elif config["algorithm"] == "convae":
        experiment = ConvAEExperiment(config)
        experiment.run()

    elif config["algorithm"] == "pcnet":
        experiment = ExperimentPC(config)
        experiment.run()

    elif config["algorithm"] == "pcnet_ge":
        experiment = ExperimentGEPC(config)
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
        return 0.6*ce + 0.4*dl + self.gamma*(1 - cl)

    
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
            self.data = CrackDataset(self.images_path, self.ground_truth_path,sample_size=self.sample_size, resize_size=(self.resize_size[0], self.resize_size[1]))
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
                self.test_data = CrackDataset(self.test_images_path, self.test_ground_truth_path,sample_size=self.test_size, resize_size=(self.resize_size[0], self.resize_size[1]))
                print_info(f"Test samples: {len(self.test_data)}")
            self.experiment_dict["dataset_size"] = {
                "train": len(self.data),
                "val": len(self.val_data) if self.validation_split > 0 else 0,
                "test": len(self.test_data) if test_path is not None else 0
            }
        elif self.execution_mode == "test":
            self.images_path = test_path + "/images/"
            self.ground_truth_path = test_path + "/masks/"
            self.test_data = CrackDataset(self.images_path, self.ground_truth_path, augmentation_percentage=0.0,sample_size=self.test_size, resize_size=(self.resize_size[0], self.resize_size[1]))
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
    def dice_coef(self, pred, target, smooth=1.0):
        """
        pred: (B, C, H, W) logits
        target: (B, H, W) or (H, W) integer mask
        """
        if target.dim() == 2:  # single mask, no batch
            target = target.unsqueeze(0)  # (1,H,W)

        if pred.size(1) == 1:  # binary
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()  # (B,1,H,W)
        else:  # multi-class
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=pred.size(1))  # (B,H,W,C)
            target = target.permute(0,3,1,2).float()  # (B,C,H,W)

        intersection = (pred_prob * target).sum(dim=(2,3))
        denominator = pred_prob.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (denominator + smooth)
        return dice.mean()



    def iou_score(self, pred, target, smooth=1.0):
        """
        pred: (B, C, H, W) logits
        target: (B, H, W) or (H, W) integer mask
        """
        if target.dim() == 2:  # single mask
            target = target.unsqueeze(0)  # (1,H,W)

        if pred.size(1) == 1:  # binary segmentation
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()  # (B,1,H,W)
        else:  # multi-class segmentation
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=pred.size(1))  # (B,H,W,C)
            target = target.permute(0,3,1,2).float()  # (B,C,H,W)

        intersection = (pred_prob * target).sum(dim=(2,3))
        union = pred_prob.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    def cl_iou_score(self,pred_mask, true_mask, tau=2):
        """
        pred_mask, true_mask: (B,1,H,W)
        """
        B = pred_mask.shape[0]
        device = pred_mask.device
        pred_mask = (pred_mask > 0.5).float()
        true_mask = (true_mask > 0.5).float()

        # Skeletonization approximation
        kernel_3 = torch.ones((1, 1, 3, 3), device=device)
        eroded_true = F.max_pool2d(1 - true_mask, 3, 1, 1)
        eroded_pred = F.max_pool2d(1 - pred_mask, 3, 1, 1)
        skel_true = torch.relu(true_mask - (1 - eroded_true))
        skel_pred = torch.relu(pred_mask - (1 - eroded_pred))

        # --- Dilation ---
        k_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tau+1, 2*tau+1))
        k = torch.tensor(k_np, dtype=torch.float32, device=device)[None, None, :, :]
        dil_true = (F.conv2d(skel_true, k, padding=tau, groups=1) > 0).float()
        dil_pred = (F.conv2d(skel_pred, k, padding=tau, groups=1) > 0).float()

        TPcl = (skel_true * dil_pred).sum(dim=(1,2,3))
        FPcl = (skel_pred * (1 - dil_true)).sum(dim=(1,2,3))
        FNcl = (skel_true * (1 - dil_pred)).sum(dim=(1,2,3))

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

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_dice", train_dice, step=epoch)
            mlflow.log_metric("train_iou", train_iou, step=epoch)
            mlflow.log_metric("train_cl_iou", train_cl_iou, step=epoch)
            if val_loader:
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_dice", val_dice, step=epoch)
                mlflow.log_metric("val_iou", val_iou, step=epoch)
                mlflow.log_metric("val_cl_iou", val_cl_iou, step=epoch)
            
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
        test_loss, test_dice, test_iou, test_cliou = 0,0,0,0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
                test_loss += loss.item()
                test_dice += self.dice_coef(outputs, masks).item()
                test_iou  += self.iou_score(outputs, masks).item()
                test_cliou += self.cl_iou_score(outputs, masks)
        print(f"Test Loss: {test_loss/len(test_loader)}")
        mlflow.log_metric("test_loss", test_loss/len(test_loader))
        print(f"Test Dice: {test_dice/len(test_loader)}")
        mlflow.log_metric("test_dice", test_dice/len(test_loader))
        print(f"Test IoU: {test_iou/len(test_loader)}")
        mlflow.log_metric("test_iou", test_iou/len(test_loader))
        print(f"Test CL IoU: {test_cliou/len(test_loader)}")
        mlflow.log_metric("test_cliou", test_cliou/len(test_loader))
        with open(self.experiment_report, "a") as f:
            f.write(f"Test Loss: {test_loss/len(test_loader)}\n")
            f.write(f"Test dice: {test_dice/len(test_loader)}\n")
            f.write(f"Test iou: {test_iou/len(test_loader)}\n")
            f.write(f"Test cliou: {test_cliou/len(test_loader)}\n")
        f.close()
        return test_loss/len(test_loader)        

    def log_results(self):
        pass

    def show_predictions(self, model, dataset, device, num_samples=3, threshold=0.5):
        # Save model info
        model_json = os.path.join(self.log_path, "model_info.json")
        json.dump(model.info(), open(model_json, "w"), indent=4)
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

        for j in range(num_groups):
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

            plt.tight_layout()
            plt.savefig(os.path.join(predictions_dir, f"predictions_grid_{j}.png"))
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
            self.show_predictions(self.model, self.test_data, self.device, num_samples=2, threshold=self.threshold)

class UNetExperiment(ExperimentDL):

    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unet")

    def set_algorithm(self):
        self.model = UNet(n_classes=1)
    
    def save_model(self, model_name="unet"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))
        mlflow.pytorch.log_model(self.model, artifact_path=model_name)

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
        mlflow.pytorch.log_model(self.model, artifact_path=model_name)

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
        mlflow.pytorch.log_model(self.model, artifact_path=model_name)

    def load_model(self, model_name="unetplus"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentPC(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unet")
        self.input_channels = self.config.get("input_channels", 3)
        self.max_depth = self.config.get("max_depth", 5)
        self.max_branching = self.config.get("max_branching", 2)
        self.n_classes = self.config.get("n_classes", 2)
        self.input_nodes = []

        self.epochs = self.config.get("epochs", 10)
        self.cv_epochs = self.config.get("cv_epochs", 5)

        self.seed = self.config.get("seed", 42)
        self.feature_extractor = self.config.get("feature_extractor", "handcrafted")
        self.filter_sizes = self.config.get("filter_sizes", (3, 7, 15))
        self.n_cpus = self.config.get("n_cpus", None)
        self.batch_size = self.config.get("batch_size", 4)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        self.n_cv_modules = self.config.get("n_cv_modules", 32)
        self.n_filters = self.config.get("n_filters", 4)
        self.set_algorithm()
        self._setup_cpu_env()
        self.cv_factory = None


        # cache for conv filter bank tensor
        self.cached_filter_tensor = None

    # --------------------------------------------------------
    # Algorithm
    # --------------------------------------------------------
    def set_algorithm(self):
        self.model = PCNet(
            n_classes=self.n_classes,
            distribution=None,
            device="cpu",
            max_depth=self.max_depth,
            max_branching=self.max_branching,
            seed=self.seed,
            cv_module = None
        )

    # --------------------------------------------------------
    # Threading & environment setup
    # --------------------------------------------------------
    def _setup_cpu_env(self):
        if self.n_cpus is None or self.n_cpus < 1:
            self.n_cpus = multiprocessing.cpu_count()
        # unify threads across libs
        torch.set_num_threads(self.n_cpus)
        os.environ["OMP_NUM_THREADS"] = str(self.n_cpus)
        os.environ["MKL_NUM_THREADS"] = str(self.n_cpus)
        print(f"⚙️ Using {self.n_cpus} CPU cores (torch threads)")

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------
    def get_filter_bank(self):
        """Comprehensive crack-oriented filter bank."""
        filters = {
            # Basic edge filters
            "sobel_x": np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], np.float32),
            "sobel_y": np.array([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], np.float32),
            "laplacian": np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], np.float32),
            "sharpen": np.array([[ 0, -1,  0],
                                 [-1,  5, -1],
                                 [ 0, -1,  0]], np.float32),
            "gauss3": (cv2.getGaussianKernel(3, 0.8) @
                       cv2.getGaussianKernel(3, 0.8).T).astype(np.float32),
        }

        # Directional Gabor filters
        for theta in np.linspace(0, np.pi, 6, endpoint=False):
            gabor = cv2.getGaborKernel((9, 9), 3.0, theta, 5.0, 0.5, 0, ktype=cv2.CV_32F)
            filters[f"gabor_{int(theta * 180 / np.pi)}"] = gabor

        return filters

    # --------------------------------------------------------
    # Conv filter tensor building and caching
    # --------------------------------------------------------
    def _build_filter_tensor_once(self, filters, device="cpu"):
        if self.cached_filter_tensor is not None:
            return self.cached_filter_tensor

        filt_list = list(filters.values())
        max_h = max(k.shape[0] for k in filt_list)
        max_w = max(k.shape[1] for k in filt_list)

        def pad_to(k, H, W):
            out = np.zeros((H, W), np.float32)
            h, w = k.shape
            y0, x0 = (H - h) // 2, (W - w) // 2
            out[y0:y0 + h, x0:x0 + w] = k
            return out

        filt_padded = [pad_to(k, max_h, max_w) for k in filt_list]
        filt_tensor = torch.from_numpy(np.stack(filt_padded)).float().unsqueeze(1)
        filt_tensor = filt_tensor.repeat(1, 3, 1, 1).to(device)
        self.cached_filter_tensor = filt_tensor
        return self.cached_filter_tensor

    # --------------------------------------------------------
    # Convolutional feature computation
    # --------------------------------------------------------
    def compute_conv_features(self, images, filters, device="cpu"):
        if isinstance(images, list):
            images = torch.from_numpy(np.stack(images)).float()
        elif isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # Normalize to BCHW (3 channels)
        if images.ndim != 4:
            raise ValueError(f"Expected 4D images, got {images.shape}")
        if images.shape[1] != 3 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] not in (1, 3):
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            elif images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected image shape: {tuple(images.shape)}")

        images = images.to(device)
        filt_tensor = self._build_filter_tensor_once(filters, device=device)
        conv = F.conv2d(images, filt_tensor, padding="same")  # (B, K, H, W)
        return conv

    # --------------------------------------------------------
    # Multi-scale pooling
    # --------------------------------------------------------
    def multi_scale_pooling(self, feat_maps, pool_sizes=(3, 7, 15), modes=("avg", "max")):
        pooled = []
        for k in pool_sizes:
            if k <= 1:
                continue
            for mode in modes:
                if mode == "avg":
                    out = F.avg_pool2d(feat_maps, kernel_size=k, stride=1, padding=k // 2)
                elif mode == "max":
                    out = F.max_pool2d(feat_maps, kernel_size=k, stride=1, padding=k // 2)
                else:
                    raise ValueError(f"Unknown pooling mode: {mode}")
                pooled.append(out)
        return torch.cat(pooled, dim=1) if pooled else feat_maps

    # --------------------------------------------------------
    # Handcrafted features
    # --------------------------------------------------------
    def get_handcrafted_features(self, images, n_jobs=8, prune_features_idx=None):
        """Compute handcrafted features for images in parallel."""
        def process_one(img):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0

            # Normalize to HWC
            if img.ndim == 2:
                img_hwc = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
                img_hwc = np.transpose(img, (1, 2, 0))
            elif img.ndim == 3 and img.shape[-1] == 3:
                img_hwc = img
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            gray = cv2.cvtColor((img_hwc * 255).astype(np.uint8),
                                cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            lab = cv2.cvtColor((img_hwc * 255).astype(np.uint8),
                               cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            hsv = cv2.cvtColor((img_hwc * 255).astype(np.uint8),
                               cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0

            feats = [
                gray,
                lab[..., 0], lab[..., 1], lab[..., 2],
                hsv[..., 0], hsv[..., 1], hsv[..., 2],
                cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1),
                cv2.Laplacian(gray, cv2.CV_32F),
            ]

            # Gabor filters (4 orientations)
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                kernel = cv2.getGaborKernel((9, 9), 2.0, theta, 4.0, 0.5, 0, ktype=cv2.CV_32F)
                feats.append(cv2.filter2D(gray, cv2.CV_32F, kernel))

            # Optional Frangi (line enhancement)
            # if self.use_frangi:
            frangi_resp = frangi(gray, sigmas=(1, 5), scale_step=2,
                                    beta=0.5)
            feats.append(frangi_resp.astype(np.float32))

            # Optional High-pass
            # if self.use_highpass:
            blurred = cv2.GaussianBlur(gray, (0, 0), 5)
            highpass = cv2.subtract(gray, blurred)
            feats.append(highpass.astype(np.float32))

            return np.stack(feats, axis=0).astype(np.float32)

        workers = min(n_jobs if n_jobs else multiprocessing.cpu_count(), 16)
        print(f"⚙️ Using {workers} CPU workers for handcrafted feature extraction...")
        feats_list = Parallel(n_jobs=workers, backend="loky", verbose=10)(
            delayed(process_one)(img) for img in images
        )
        feats = np.stack(feats_list, axis=0)  # (B, C, H, W)

        # Conv bank (batched)
        conv_feats = self.compute_conv_features(images, self.get_filter_bank(), device=self.device)
        conv_feats = conv_feats.detach().cpu().numpy()

        base_feats = np.concatenate([feats, conv_feats], axis=1)
        base_tensor = torch.from_numpy(base_feats).float().to(self.device)

        pooled_feats = self.multi_scale_pooling(base_tensor,
                                                pool_sizes=self.filter_sizes,
                                                modes=("avg", "max"))
        pooled_feats = pooled_feats.cpu().numpy()
        final_feats = np.concatenate([base_feats, pooled_feats], axis=1)
        if prune_features_idx is not None:
            final_feats = final_feats[:, prune_features_idx, :, :]

        return final_feats

    # TODO one process per module
    def eval_cv_pop(self, modules, images):
        with multiprocessing.Pool(processes=self.n_cpus) as pool:
            results = pool.starmap(
                self._worker_process_wrapper,
                [(idx, module, images) for idx, module in enumerate(modules)]
            )

        return [fit for fit, _ in results]

    def _worker_process_wrapper(self, idx, module, imgs):
        
        print_debugging(f"[{time.strftime('%H:%M:%S')}] Worker {idx} starting...")
        try:
            feats = [module.get_output(img, return_map=True) for img in imgs]
            var = np.stack(feats).var()
            return (float(var), idx)
        except Exception as e:
            print(f"[Worker {idx}] error: {e}")
            return (float("-inf"), idx)

    def convolution_modules_tensor(
        self, images, cv_factory, return_maps=True, n_jobs=None, verbose=0, prune_features=None
    ):
        """
        Torchrun-friendly version with multiprocessing image parallelism.
        - Each rank processes a disjoint shard of images.
        - Each shard is processed in parallel using multiprocessing (per-image chunks).
        - Results are gathered with torch.distributed all_gather on tensors.
        - Output shape: [N_images, C, H, W] or [N_images, H, W] depending on your modules.
        """
        import multiprocessing as mp
        from joblib import Parallel, delayed
        import torch.distributed as dist
        import numpy as np
        import os

        # ---------------------------------------------------------
        # Normalize input to numpy
        # ---------------------------------------------------------
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if images.ndim == 3:
            images = np.expand_dims(images, 0)  # [1, H, W]
        total_images = len(images)
        

        # ---------------------------------------------------------
        # Optimize module via variance (if no pruning provided)
        # ---------------------------------------------------------
        if prune_features is None:
            # ---------------------------------------------------------
            # CMA-ES optimization loop
            # ---------------------------------------------------------
            # Multiprocessing parallelization over MODULES
            modules = cv_factory.ask_pop()
            n_procs = min(n_jobs, len(modules))
            print_info(f"⚙️ Using {n_procs} processes for convolution module evaluation...")
            for epoch in trange(self.cv_epochs, desc="CV Module Optimization", disable=verbose == 0):
                print_debugging(f"Epoch {epoch}")
                modules = cv_factory.ask_pop()
                if not isinstance(modules, list):
                    modules = [modules]               

                fitness = self.eval_cv_pop(modules, images)

                print_debugging(f"Module fitness values: {fitness}")
                # CMA-ES update
                cv_factory.tell_pop(fitness)

        # ---------------------------------------------------------
        # Best module and final extraction
        # ---------------------------------------------------------
        module = cv_factory.best_module()

        feats_np = [module.get_output(img, return_map=return_maps) for img in images]
        feats_np = np.stack(feats_np, axis=0)  # [N, C, H, W] or [N, H, W]

        # ---------------------------------------------------------
        # Feature pruning and normalization
        # ---------------------------------------------------------
        if prune_features is not None:
            if feats_np.ndim == 4:  # [N, C, H, W]
                feats_np = feats_np[:, prune_features, :, :]
            else:
                feats_np = feats_np[prune_features, :, :]

        print_debugging(
            f"Convolution module features shape: {feats_np.shape}, "
            f"value range: {feats_np.min()} to {feats_np.max()}, var: {feats_np.var()}"
        )
        feats_np = (feats_np - feats_np.mean()) / (feats_np.std() + 1e-8)
        return feats_np
    
    def get_cv_module(
        self, images, cv_factory, return_maps=True, n_jobs=None, verbose=0, prune_features=None
    ):
        """
        Torchrun-friendly version with multiprocessing image parallelism.
        - Each rank processes a disjoint shard of images.
        - Each shard is processed in parallel using multiprocessing (per-image chunks).
        - Results are gathered with torch.distributed all_gather on tensors.
        - Output shape: [N_images, C, H, W] or [N_images, H, W] depending on your modules.
        """
        import multiprocessing as mp
        from joblib import Parallel, delayed
        import torch.distributed as dist
        import numpy as np
        import os

        # ---------------------------------------------------------
        # Normalize input to numpy
        # ---------------------------------------------------------
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if images.ndim == 3:
            images = np.expand_dims(images, 0)  # [1, H, W]
        total_images = len(images)
        

        # ---------------------------------------------------------
        # Optimize module via variance (if no pruning provided)
        # ---------------------------------------------------------
        if prune_features is None:
            # ---------------------------------------------------------
            # CMA-ES optimization loop
            # ---------------------------------------------------------
            # Multiprocessing parallelization over MODULES
            modules = cv_factory.ask_pop()
            n_procs = min(n_jobs, len(modules))
            print_info(f"⚙️ Using {n_procs} processes for convolution module evaluation...")
            for epoch in trange(self.cv_epochs, desc="CV Module Optimization", disable=verbose == 0):
                print_debugging(f"Epoch {epoch}")
                modules = cv_factory.ask_pop()
                if not isinstance(modules, list):
                    modules = [modules]               

                fitness = self.eval_cv_pop(modules, images)

                print_debugging(f"Module fitness values: {fitness}")
                # CMA-ES update
                cv_factory.tell_pop(fitness)

        # ---------------------------------------------------------
        # Best module and final extraction
        # ---------------------------------------------------------
        module = cv_factory.best_module()

        return module



    def get_feature_tensor(self, images, prune_features=None):
        if self.feature_extractor == "handcrafted":
            return self.get_handcrafted_features(images, n_jobs=self.n_cpus, prune_features_idx=prune_features)
        elif self.feature_extractor == "cv_modules":
            if self.cv_factory is None:
                self.cv_factory = CVModuleFactory(
                    Optimizer={"class_name": "CMAES", "kwargs": {"n_params": 128, "lambda_": self.n_cv_modules, "seed": self.seed}},
                    CVModule={"class_name": "ConvolutionModule", "kwargs": {"n_filters": self.n_filters, "filter_sizes": 3, "device": self.device}}
                )
            return self.convolution_modules_tensor(images, self.cv_factory, return_maps=True, n_jobs=self.n_cpus, prune_features=prune_features)

        else:
            raise ValueError(f"Unknown feature extractor {self.feature_extractor}")
    
    def init_cv_module(self, images):
        self.cv_factory = CVModuleFactory(
                    Optimizer={"class_name": "CMAES", "kwargs": {"n_params": 128, "lambda_": self.n_cv_modules, "seed": self.seed}},
                    CVModule={"class_name": "ConvolutionModule", "kwargs": {"n_filters": self.n_filters, "filter_sizes": 3, "device": self.device}}
                )
        return self.get_cv_module(images, self.cv_factory, return_maps=True, n_jobs=self.n_cpus)
        
    def set_criterion(self, target):
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        pos_weight = self.compute_pos_weight(target)
        self.criterion = DiceCELoss(pos_weight=pos_weight.float().to(self.device))
    # --------------------------------------------------------
    # Training & Evaluation
    # --------------------------------------------------------
    def train(self, model, epochs=10, lr=1e-3, n_cpus=None):
        # Unify threading
        self._setup_cpu_env()

        optimizer = torch.optim.Adam(model.params, lr=lr, weight_decay=1e-4)
        print_info(f"Number of trainable parameters: {sum(p.numel() for p in model.params)}")

        feature_tensor = torch.from_numpy(self.feature_tensor).float()
        target = torch.from_numpy(self.train_masks).long()
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        # DataLoaders
        train_loader = self.get_dataloaders(TensorDataset(feature_tensor, target), shuffle=True, batch_size=self.batch_size)
        val_loader = None

        if self.val_feature_tensor is not None and self.val_masks is not None:
            val_feature_tensor = torch.from_numpy(self.val_feature_tensor).float()
            val_target = torch.from_numpy(self.val_masks).long()
            if val_target.dim() == 4 and val_target.size(1) == 1:
                val_target = val_target.squeeze(1)
            val_loader = self.get_dataloaders(
                TensorDataset(val_feature_tensor, val_target),
                shuffle=False,
                batch_size=self.batch_size
            )

        # Logs
        train_losses, val_losses = [], []
        train_dices, val_dices = [], []
        train_ious, val_ious = [], []
        train_clious, val_clious = [], []
        gates_over_epochs = []

        pbar = tqdm.tqdm(range(epochs), desc="Training PCNet")
        for epoch in pbar:
            batch_losses, batch_dices, batch_ious, batch_clious = [], [], [], []

            for ft_batch, t_batch in train_loader:
                optimizer.zero_grad()
                logits = model.evaluate(ft_batch)                     # (B, C, H, W) log-probs
                loss = self.criterion(logits, t_batch)
                loss.backward()
                optimizer.step()

                # Metrics (vectorized)
                pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()   # (B,1,H,W)
                true_mask = t_batch.unsqueeze(1).float()                        # (B,1,H,W)

                dice_score = self.dice_coef(logits, t_batch)      # tensor
                iou_score = self.iou_score(logits, t_batch)       # tensor
                cl_iou_score = self.cl_iou_score(pred_mask, true_mask)  # float or tensor

                batch_losses.append(loss.item())
                batch_dices.append(float(torch.as_tensor(dice_score).detach().mean()))
                batch_ious.append(float(torch.as_tensor(iou_score).detach().mean()))
                batch_clious.append(float(torch.as_tensor(cl_iou_score).detach()))


            # Collect gate values (if present) in parallel
            gates_values = []
            for n in model.get_nodes():
                if hasattr(n, "gate"):
                    with torch.no_grad():
                        gates_values.append(float(n.gate.item()))
            if gates_values:
                gates_over_epochs.append(gates_values)

            # Epoch metrics
            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            epoch_dice = float(np.mean(batch_dices)) if batch_dices else 0.0
            epoch_iou = float(np.mean(batch_ious)) if batch_ious else 0.0
            epoch_cl_iou = float(np.mean(batch_clious)) if batch_clious else 0.0

            train_losses.append(epoch_loss)
            train_dices.append(epoch_dice)
            train_ious.append(epoch_iou)
            train_clious.append(epoch_cl_iou)

            

            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_dice": epoch_dice,
                "train_iou": epoch_iou,
                "train_cl_iou": epoch_cl_iou
            }, step=epoch)

            # Validation
            if val_loader is not None:
                val_loss, val_dice, val_iou, val_cl_iou = self.evaluate(model, val_loader)
                val_losses.append(val_loss)
                val_dices.append(val_dice)
                val_ious.append(val_iou)
                val_clious.append(val_cl_iou)
                mlflow.log_metrics({
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_cl_iou": val_cl_iou
                }, step=epoch)

            if epochs % 5 == 0:
                with open(self.experiment_report, "a") as f:
                    f.write(f"{epoch}\ttrain\t{epoch_loss}\t{epoch_cl_iou}\n")
                    f.write(f" \tval\t{val_loss}\t{val_cl_iou}\n")

            pbar.set_postfix({
                "train_loss": f"{epoch_loss:.4f}",
                "val_loss": f"{val_losses[-1]:.4f}" if val_loader and val_losses else None
            })

        # Save logs
        df_train = pd.DataFrame({
            "train_loss": train_losses,
            "train_dice": train_dices,
            "train_iou": train_ious,
            "train_cl_iou": train_clious
        })
        df_train.to_csv(os.path.join(self.log_path, "training.csv"), index=False)

        if val_loader is not None:
            df_val = pd.DataFrame({
                "val_loss": val_losses,
                "val_dice": val_dices,
                "val_iou": val_ious,
                "val_cl_iou": val_clious
            })
            df_val.to_csv(os.path.join(self.log_path, "validation.csv"), index=False)

        # Save gates (if any)
        if gates_over_epochs:
            gates = np.array([np.pad(g, (0, max(map(len, gates_over_epochs)) - len(g)), constant_values=np.nan)
                              for g in gates_over_epochs], dtype=np.float32)
            df_importance = pd.DataFrame(gates, columns=[f"gate_{i}" for i in range(gates.shape[1])])
            df_importance.to_csv(os.path.join(self.log_path, "gates.csv"), index=False)

        # Plots
        import matplotlib.pyplot as plt
        # Loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        if val_loader:
            plt.plot(val_losses, label="Val Loss")
        plt.legend(); plt.title("Training Loss over Epochs")
        plt.savefig(os.path.join(self.log_path, "training_plot.png")); plt.close()

        # Dice
        plt.figure(figsize=(8, 5))
        plt.plot(train_dices, label="Train Dice")
        if val_loader:
            plt.plot(val_dices, label="Val Dice")
        plt.legend(); plt.title("Training Dice over Epochs")
        plt.savefig(os.path.join(self.log_path, "dice_plot.png")); plt.close()

        # IoU
        plt.figure(figsize=(8, 5))
        plt.plot(train_ious, label="Train IoU")
        if val_loader:
            plt.plot(val_ious, label="Val IoU")
        plt.legend(); plt.title("Training IoU over Epochs")
        plt.savefig(os.path.join(self.log_path, "iou_plot.png")); plt.close()

        # clIoU
        plt.figure(figsize=(8, 5))
        plt.plot(train_clious, label="Train CL IoU")
        if val_loader:
            plt.plot(val_clious, label="Val CL IoU")
        plt.legend(); plt.title("Training CL IoU over Epochs")
        plt.savefig(os.path.join(self.log_path, "cliou_plot.png")); plt.close()

        print_info("✅ Training complete.")
        with open(self.experiment_report, "a") as f:
            f.write(f"Final Training Loss: {train_losses[-1]}, Dice: {train_dices[-1]}, IoU: {train_ious[-1]}, CL IoU: {train_clious[-1]}\n")
            if val_loader:
                f.write(f"Final Validation Loss: {val_losses[-1]}, Dice: {val_dices[-1]}, IoU: {val_ious[-1]}, CL IoU: {val_clious[-1]}\n")

        return model

    @torch.no_grad()
    def evaluate(self, model, dataloader):
        losses, dices, ious, clious = [], [], [], []
        for ft_batch, t_batch in dataloader:
            logits = model.evaluate(ft_batch)
            loss = self.criterion(logits, t_batch)
            dice_score = self.dice_coef(logits, t_batch)
            iou_score = self.iou_score(logits, t_batch)
            pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
            true_mask = t_batch.unsqueeze(1).float()
            cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

            losses.append(loss.item())
            dices.append(float(torch.as_tensor(dice_score).mean()))
            ious.append(float(torch.as_tensor(iou_score).mean()))
            clious.append(float(torch.as_tensor(cl_iou_score)))
        return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious)), float(np.mean(clious))

    # --------------------------------------------------------
    # I/O
    # --------------------------------------------------------
    def show_prediction(self, model):
        with open(self.experiment_report, "a") as f:
            f.write(str(model) + "\n")

        predictions_path = os.path.join(self.log_path, "predictions")
        os.makedirs(predictions_path, exist_ok=True)

        imgs = [self.test_data[i][0] for i in range(len(self.test_data))]

        # Process images in groups of 3
        group_size = 3
        for start_idx in range(0, len(self.test_inputs), group_size):
            end_idx = min(start_idx + group_size, len(self.test_inputs))
            group = list(zip(
                imgs[start_idx:end_idx],
                self.test_inputs[start_idx:end_idx],
                self.test_ground_truth[start_idx:end_idx]
            ))

            fig, axes = plt.subplots(3, group_size, figsize=(4 * group_size, 9))

            for col, (img, inputs, gt) in enumerate(group):
                mask = model.predict(inputs)
                img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
                gt_np = gt.squeeze().cpu().numpy()
                mask_np = mask[0][0] if isinstance(mask, (list, tuple)) else mask.squeeze()

                # Row 1 → Input
                axes[0, col].imshow(img_np)
                axes[0, col].set_title(f"Input {start_idx + col}")
                axes[0, col].axis("off")

                # Row 2 → Ground Truth
                axes[1, col].imshow(gt_np, cmap="gray")
                axes[1, col].set_title(f"Ground Truth {start_idx + col}")
                axes[1, col].axis("off")

                # Row 3 → Prediction
                axes[2, col].imshow(mask_np, cmap="cool")
                axes[2, col].set_title(f"Prediction {start_idx + col}")
                axes[2, col].axis("off")

            # Hide any unused columns (in case last group < 3)
            for col in range(len(group), group_size):
                axes[0, col].axis("off")
                axes[1, col].axis("off")
                axes[2, col].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(predictions_path, f"predictions_{start_idx//group_size}.png"))
            plt.close(fig)

        print_debugging(self.log_path)
        model.visualize(save_path=self.log_path)

    def finetune(self):
        pass

    def save_model(self, model_name="pcnet"):
        with open(os.path.join(self.log_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.save_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, model_name="pcnet"):
        with open(os.path.join(self.save_path, f"{model_name}.pkl"), "rb") as f:
            self.model = pickle.load(f)

    # --------------------------------------------------------
    # Orchestration
    # --------------------------------------------------------
    def run(self):
        self._setup_cpu_env()
        self.set_dataset(self.data_path, self.test_path)
        print_info("Dataset size:", len(self.data))
        if self.execution_mode == "train":
            self.set_algorithm()
            self.feature_tensor = self.get_feature_tensor(np.array([self.train_data[i][0] for i in range(len(self.train_data))]))
            self.train_masks = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
            self.val_feature_tensor = self.get_feature_tensor(np.array([self.val_data[i][0] for i in range(len(self.val_data))]))
            self.val_masks = np.array([self.val_data[i][1] for i in range(len(self.val_data))]) if self.validation_split > 0 else None
            self.set_criterion(torch.from_numpy(self.train_masks).float().to(self.device))

            self.model.init_network(self.feature_tensor, self.train_masks)
            with open(self.experiment_report, "a") as f:
                f.write(str(self.model) + "\n")
            if self.device == "cuda":
                self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            else:
                self.model = self.train_parallel(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            self.save_model()

            if self.test_path is not None:
                self.test_inputs = torch.from_numpy(self.get_feature_tensor(np.array([self.test_data[i][0] for i in range(len(self.test_data))]))).float().to(self.device)
                self.test_ground_truth = torch.from_numpy(np.array([self.test_data[i][1] for i in range(len(self.test_data))])).long().to(self.device)
                if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)

                test_loader = self.get_dataloaders(TensorDataset(self.test_inputs, self.test_ground_truth), shuffle=False, batch_size=self.batch_size)
                loss, dice, iou, cl_iou = self.evaluate(self.model, test_loader)
                print_info("Test Loss:", loss, "Dice:", dice, "IoU:", iou, "CL IoU:", cl_iou)
                with open(self.experiment_report, "a") as f:
                    f.write(f"Test Loss: {loss}, Dice: {dice}, IoU: {iou}, CL IoU: {cl_iou}\n")
                self.show_prediction(self.model)
        elif self.execution_mode == "test":
            self.load_model(self.model_name)
            self.test_inputs = torch.from_numpy(self.get_feature_tensor(np.array([self.test_data[i][0] for i in range(len(self.test_data))]))).float().to(self.device)
            self.test_ground_truth = torch.from_numpy(np.array([self.test_data[i][1] for i in range(len(self.test_data))])).long().to(self.device)
            if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                self.test_ground_truth = self.test_ground_truth.squeeze(1)
            self.show_prediction(self.model)


class ExperimentGEPC(ExperimentPC):
    def __init__(self, config):
        super().__init__(config)
        self.ge_config = self.config.get("ge", {"generations": 10, "pop_size": 8, "genome_len": 96, "crossover_rate": 0.9, "mutation_rate": 0.05, "elite_k": 2, "train_split": 0.05})
        self.ge = GEOptimizer(
                    pop_size=self.ge_config["pop_size"],
                    genome_len=self.ge_config["genome_len"],
                    max_depth=self.max_depth,          # search budget
                    max_branching=self.max_branching,      # upper bound for sum fan-in
                    crossover_rate=self.ge_config["crossover_rate"],
                    mutation_rate=self.ge_config["mutation_rate"],
                    elite_k=self.ge_config["elite_k"],
                    n_classes=self.n_classes,
                    device=self.device,
                    n_jobs=self.n_cpus,
                    report=self.experiment_report
                )
        
    def get_prune_features_idx(self, model):
        feats = []
        new_name = 0
        for n in model.get_nodes():
            if isinstance(n, InputNode):
                feats.append(n.feature_idx)
                n.feature_name = f"feat_{new_name}"
                n.feature_idx = new_name
                new_name += 1

        return feats
    
    def worker_cv_module_process(self, model, module, X_np, Y_t, device, prune_idx, queue):
        """
        Worker that evaluates one or more modules on all training images.
        Returns list of fitness values (one per module).
        """
            
        # Extract features for this module
        feats = []
        for img in X_np:
            out = module.get_output(img, return_map=True)
            feats.append(torch.from_numpy(out))
        X_tensor = torch.stack(feats).float().to(device)

        if prune_idx is not None:
            X_tensor = X_tensor[:, prune_idx, :, :]
    
        dataset = TensorDataset(X_tensor, Y_t)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        loss_total, cliou_total = 0.0, 0.0
        with torch.no_grad():
            for Xb, Yb in loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                logits = model.evaluate(Xb)
                loss_total += float(self.criterion(logits, Yb.squeeze(1)).item())
                cliou_total += float(
                    self.cl_iou_score(
                        torch.argmax(logits, dim=1, keepdim=True).float(), Yb
                    )
                )
        avg_loss = loss_total / max(1, len(loader))
        return avg_loss
    
    def train_cv_modules(self, model, X_train, Y_train, prune_features=None, epochs=10, lr=1e-2, n_cpus=8):
        self.cv_factory._optimizer.best = None  # reset best found solution
        self.cv_factory._optimizer.best_fitness = -np.inf  # reset best found solution

        for epoch in trange(epochs, desc="Training CV Modules"):
            print_info(f"CV Modules Training Epoch {epoch + 1}/{epochs}")
            modules = self.cv_factory.ask_pop()
            if not isinstance(modules, list):
                modules = [modules]

            # Split modules into chunks for each worker
            n_procs = min(n_cpus, len(modules))
            with multiprocessing.Pool(n_procs) as pool:
                fitnesses = pool.starmap(
                    self.worker_cv_module_process,
                    [(model, module, X_train, Y_train, self.device, prune_features, None) for module in modules]
                )
            print_info(f"   CV Module Fitnesses: {fitnesses}")
            self.cv_factory.tell_pop(fitnesses)

        return self.cv_factory.best_module()
        
    def run(self):
        
        self.set_dataset(self.data_path, self.test_path)
        print_info("Dataset size:", len(self.data))
        if self.execution_mode == "train":
            self.set_algorithm()
            

            ge_train_size = int(self.ge_config["train_split"] * len(self.train_data))
            print_info("GE Training size:", ge_train_size)
            print_info(len(self.train_data))
            ge_data = []
            ge_masks = []
            for i in range(ge_train_size):
                try:
                    x = self.train_data[i]
                    ge_data.append(x[0])
                    ge_masks.append(x[1])
                except Exception as e:
                    print_error(f"Error processing training sample {i}: {e}")
            ge_X_train = np.array(ge_data)
            ge_masks = np.array(ge_masks)
            ge_X_train = torch.from_numpy(ge_X_train).float().to(self.device)
            ge_Y_train = torch.from_numpy(ge_masks).long().to(self.device)
            self.set_criterion(ge_Y_train)
            best_pc, best_genome, history = self.ge.evolve(ge_X_train, ge_Y_train, criterion=self.criterion, generations = self.ge_config["generations"], n_classes=self.n_classes, train_steps=50, lr=self.learning_rate)
            print_info("Best genome:", best_genome)
            print_info("Best PC:", str(best_pc))
            with open(self.experiment_report, "a") as f:
                f.write(str(best_pc) + "\n")
                f.write(f"History: {history}\n")
                f.write(f"Best genome: {best_genome}\n")

            self.model = best_pc
            f.close()
            self.model.cv_module = None
            train_data = []
            train_masks = []
            for i in range(len(self.train_data)):
                try:
                    x = self.train_data[i]
                    train_data.append(x[0])
                    train_masks.append(x[1])
                except Exception as e:
                    print_error(f"Error processing training sample {i}: {e}")

            self.feature_tensor = np.array(train_data)
            self.train_masks = np.array(train_masks)

            val_data = []
            val_masks = []
            for i in range(len(self.val_data)):
                try:
                    x = self.val_data[i]
                    val_data.append(x[0])
                    val_masks.append(x[1])
                except Exception as e:
                    print_error(f"Error processing validation sample {i}: {e}")

            self.val_feature_tensor = np.array(val_data)
            self.val_masks = np.array(val_masks)
            self.set_criterion(torch.from_numpy(self.train_masks).long().to(self.device))
            if self.device == "cuda":
                self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            else:
                self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            self.save_model("pcnet-ge")
            if self.test_path is not None:
                test_data = []
                test_masks = []
                for i in range(len(self.test_data)):
                    try:
                        x = self.test_data[i]
                        test_data.append(x[0])
                        test_masks.append(x[1])
                    except Exception as e:
                        print_error(f"Error processing validation sample {i}: {e}")
                self.test_inputs = torch.from_numpy(np.array(test_data)).float().to(self.device)
                self.test_ground_truth = torch.from_numpy(np.array(test_masks)).long().to(self.device)
                if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)
                
                test_loader = self.get_dataloaders(TensorDataset(self.test_inputs, self.test_ground_truth), shuffle=False, batch_size=self.batch_size)
                loss, dice, iou, cl_iou = self.evaluate(self.model, test_loader)
                print_info("Test Loss:", loss, "Dice:", dice, "IoU:", iou, "CL IoU:", cl_iou)
                with open(self.experiment_report, "a") as f:
                    f.write(f"Test Loss: {loss}, Dice: {dice}, IoU: {iou}, CL IoU: {cl_iou}\n")
                f.close()
                self.show_prediction(self.model)
        elif self.execution_mode == "test":
            self.load_model(self.model_name)
            self.test_inputs = torch.from_numpy(self.get_feature_tensor(np.array([self.test_data[i][0] for i in range(len(self.test_data))]))).float().to(self.device)
            self.test_ground_truth = torch.from_numpy(np.array([self.test_data[i][1] for i in range(len(self.test_data))])).long().to(self.device)
            if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)
            self.show_prediction(self.model)
