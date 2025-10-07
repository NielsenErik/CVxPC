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

from joblib import Parallel, delayed
import json

import pandas as pd
from crackdata_processing import CrackDataset, CrackAugmentations



def run_experiment(config):
    if config["algorithm"] == "unet":
        experiment = UNetExperiment(config)
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
        self.sample_size = self.config.get("sample_size", 20)
        self.test_size = self.config.get("test_size", 10)
        self.resize_size = self.config.get("resize_size", [256, 256])
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.threshold = self.config.get("threshold", 0.5)
        
        self.epochs = self.config.get("epochs", 10)
        self.model = None

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
        self.images_path = train_path + "/images/"
        self.ground_truth_path = train_path + "/masks/"        
        self.data = CrackDataset(self.images_path, self.ground_truth_path, augmentation_percentage=0.1,sample_size=self.sample_size, resize_size=(self.resize_size[0], self.resize_size[1]))     
        if self.execution_mode == "train":
            if self.validation_split > 0:
                split_idx = int((1 - self.validation_split) * len(self.data))
                self.train_data = torch.utils.data.Subset(self.data, list(range(0, split_idx)))
                self.val_data = torch.utils.data.Subset(self.data, list(range(split_idx, len(self.data))))
                print_info(f"Training samples: {len(self.train_data)}, Validation samples: {len(self.val_data)}")
            else:
                self.train_data = self.data
                self.val_data = None
                print_info(f"Training samples: {len(self.train_data)}")
        if test_path is not None:
            self.test_images_path = test_path + "/images/"
            self.test_ground_truth_path = test_path + "/masks/"
            self.test_data = CrackDataset(self.test_images_path, self.test_ground_truth_path, augmentation_percentage=0.0,sample_size=self.test_size, resize_size=(self.resize_size[0], self.resize_size[1]))
            print_info(f"Test samples: {len(self.test_data)}")
        self.experiment_dict["dataset_size"] = {
            "train": len(self.data),
            "val": len(self.val_data) if self.validation_split > 0 else 0,
            "test": len(self.test_data) if test_path is not None else 0
        }
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


    def get_dataloaders(self, data, labels=None):
        if labels is not None:
            dataset = TensorDataset(data, labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return dataloader
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return dataloader

    @abc.abstractmethod
    def setup(self, pos_weight=1.0):
        pass

    @abc.abstractmethod
    def compute_pos_weight(self, dataset):
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

    def setup(self, pos_weight=1.0):
        self.set_optimizer(torch.optim.Adam(self.model.parameters(), lr=self.learning_rate))
        self.set_scheduler(torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1))
        self.set_loss(BCEDiceLoss(pos_weight=pos_weight))

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_loss(self, loss):
        self.loss = loss
    
    def compute_pos_weight(self, dataset):
        """
        Computes pos_weight = num_neg / num_pos for BCEWithLogitsLoss
        from a segmentation dataset.
        """
        total_pos = 0
        total_neg = 0
        
        for _, mask in dataset:  # assuming dataset returns (image, mask)
            mask = mask > 0  # ensure binary (1 = foreground, 0 = background)
            total_pos += mask.sum().item()
            total_neg += mask.numel() - mask.sum().item()
        
        if total_pos == 0:
            raise ValueError("No positive pixels found in dataset!")
        
        pos_weight = total_neg / total_pos
        return torch.tensor([pos_weight], dtype=torch.float32)

    # ----------------------------
    # Training loop
    # ----------------------------
    def train(self, model, train_data, val_data=None, 
            loss_function=None, optimizer=None, scheduler=None, 
            device="cuda", epochs=20, batch_size=8):

        # Default loss: BCE + Dice
        if loss_function is None:
            loss_function = BCEDiceLoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader = self.get_dataloaders(train_data)
        val_loader = self.get_dataloaders(val_data) if val_data else None

        model.to(device)
        train_losses, val_losses, train_dices, train_clious, val_dices, train_ious, val_ious, val_clious = [], [], [], [], [], [], [] , []
        
        for epoch in trange(epochs):
            model.train()
            epoch_loss, epoch_dice, epoch_iou, epoch_cl_iou = 0, 0, 0, 0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)              # logits
                loss = loss_function(outputs, masks) # BCE + Dice
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_dice += self.dice_coef(outputs, masks)
                epoch_iou  += self.iou_score(outputs, masks)
                epochs_cliou = self.cl_iou_score(outputs, masks)

            train_loss = epoch_loss / len(train_loader)
            train_dice = epoch_dice / len(train_loader)
            train_iou  = epoch_iou / len(train_loader)
            train_cl_iou = epochs_cliou / len(train_loader)
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
                        val_dice += self.dice_coef(outputs, masks)
                        val_iou  += self.iou_score(outputs, masks)
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
            
        loss_df = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses, 
                                "train_dice": train_dices, "val_dice": val_dices,
                                "train_iou": train_ious, "val_iou": val_ious,
                                "train_cl_iou": train_clious, "val_cl_iou": val_clious})
        
        loss_df.to_csv(os.path.join(self.log_path, "training.csv"), index=False)
        with open(self.experiment_report, "a") as f:
            f.write(f"Final Training Loss: {train_losses[-1]}, Dice: {train_dices[-1]}, IoU: {train_ious[-1]}, CL IoU: {train_clious[-1]}\n")
            if val_loader:
                f.write(f"Final Validation Loss: {val_losses[-1]}, Dice: {val_dices[-1]}, IoU: {val_ious[-1]}, CL IoU: {val_clious[-1]}\n")
        f.close()
        return model

    def finetune(self):
        pass

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
        test_loss = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
                test_loss += loss.item()
        print(f"Test Loss: {test_loss/len(test_loader)}")
        mlflow.log_metric("test_loss", test_loss/len(test_loader))
        with open(self.experiment_report, "a") as f:
            f.write(f"Test Loss: {test_loss/len(test_loader)}\n")
        f.close()
        return test_loss/len(test_loader)        

    def log_results(self):
        pass

    def show_predictions(self, model, dataset, device, num_samples=3, threshold=0.5):
        with open(self.experiment_report, "a") as f:
            f.write(model, "\n")
        f.close()
        model_json = os.path.join(self.log_path, "model_info.json")
        json.dump(model.info(), open(model_json, "w"), indent=4)

        for j in range(4):
            fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples*4))

            for i in range(num_samples):
                img, gt = dataset[i*j+i]  # one sample (no batching here)
                img = img.unsqueeze(0).to(device)  # add batch dim
                gt = gt.squeeze().cpu().numpy()

                # Prediction
                with torch.no_grad():
                    pred = model(img).cpu().squeeze().numpy()
                    prob = torch.sigmoid(torch.tensor(pred)).numpy()
                # Threshold prediction (convert probability to binary mask)
                pred_mask = (prob > threshold).astype("uint8")
                # Convert back image for plotting
                img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

                # Debug
                print("Pred:", pred.min(), pred.max(), pred.mean())
                print("Prob:", prob.min(), prob.max(), prob.mean())
                print("Pred Mask:", pred_mask.min(), pred_mask.max(), pred_mask.mean())

                # Plot: Input | Ground Truth | Prediction
                axes[i, 0].imshow(img_np)
                axes[i, 0].set_title("Input Image")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(gt, cmap="gray")
                axes[i, 1].set_title("Ground Truth")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(pred_mask, cmap="cool")
                axes[i, 2].set_title("Prediction")
                axes[i, 2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_path, f"prediction_{j}.png"))
            plt.show()

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
            pos_weight = self.compute_pos_weight(self.data)
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

class ExperimentPC(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unet")
        self.input_channels = self.config.get("input_channels", 3)
        self.max_depth = self.config.get("max_depth", 5)
        self.max_branching = self.config.get("max_branching", 2)
        self.n_classes = self.config.get("n_classes", 1)
        self.input_nodes = []
        self.epochs = self.config.get("epochs", 10)
        self.seed = self.config.get("seed", 42)
        self.feature_extractor = self.config.get("feature_extractor", "handcrafted")
        self.filter_sizes = self.config.get("filter_sizes", (3,7,15))
        self.n_cpus = self.config.get("n_cpus", 8)

    def set_algorithm(self):
        self.model = PCNet(n_classes=self.n_classes, distribution=None, device="cpu", max_depth=self.max_depth, max_branching=self.max_branching, seed=self.seed)


    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------
    def get_filter_bank(self):
        filters = {
            "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
            "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
            "laplacian": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32),
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
            # keep one small Gaussian for noise smoothing
            "gauss3": (cv2.getGaussianKernel(3, 0.8) @ cv2.getGaussianKernel(3, 0.8).T).astype(np.float32),
        }
        return filters

    def compute_conv_features(self, images, filters, device="cpu"):
        """Apply a bank of conv filters to images"""
        if isinstance(images, list):
            images = torch.from_numpy(np.stack(images)).float()
        elif isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(device)

        if images.ndim == 4 and images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2)

        filt_list = list(filters.values())
        max_h, max_w = max(k.shape[0] for k in filt_list), max(k.shape[1] for k in filt_list)

        def pad(kernel, size):
            h, w = kernel.shape
            H, W = size
            out = np.zeros((H, W), np.float32)
            out[(H - h)//2:(H - h)//2 + h, (W - w)//2:(W - w)//2 + w] = kernel
            return out

        filt_padded = [pad(k, (max_h, max_w)) for k in filt_list]
        filt_tensor = torch.from_numpy(np.stack(filt_padded)).float().unsqueeze(1)
        filt_tensor = filt_tensor.repeat(1, 3, 1, 1).to(device)

        return F.conv2d(images, filt_tensor, padding="same")
    
    def multi_scale_pooling(self, feat_maps, pool_sizes=(3, 7, 15), modes=("avg", "max")):
        """
        feat_maps: torch.Tensor (B, C, H, W)
        Returns: torch.Tensor (B, C * len(pool_sizes) * len(modes), H, W)
        """
        pooled = []
        for k in pool_sizes:
            for mode in modes:
                if mode == "avg":
                    out = F.avg_pool2d(feat_maps, kernel_size=k, stride=1, padding=k//2)
                elif mode == "max":
                    out = F.max_pool2d(feat_maps, kernel_size=k, stride=1, padding=k//2)
                else:
                    raise ValueError(f"Unknown pooling mode: {mode}")
                pooled.append(out)
        return torch.cat(pooled, dim=1)

    def get_handcrafted_features(self, images, n_jobs=8):
        """Compute handcrafted features for RGB images in parallel across CPU cores."""
        
        def process_one(img):
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
            if img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))

            C, H, W = img.shape
            feats = [img[c] for c in range(C)]
            img_hw3 = np.transpose(img, (1, 2, 0))

            gray = cv2.cvtColor((img_hw3 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            feats = [gray, sobel(gray), laplace(gray)]

            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                real, imag = gabor(gray, frequency=0.2, theta=theta)
                feats += [real, imag]

            return np.stack(feats).astype(np.float32)

        # --- Parallel computation across CPUs ---
        print(f"⚙️ Using {n_jobs} CPU workers for handcrafted feature extraction...")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_one)(img) for img in tqdm.tqdm(images, desc="Handcrafted features (parallel)"))
        
        feats = np.stack(results)  # (B, C, H, W)
        
        # --- Continue as before ---
        conv_feats = self.compute_conv_features(images, self.get_filter_bank(), device=self.device)
        base_feats = np.concatenate([feats, conv_feats], axis=1)  # (B, C_total, H, W)

        base_tensor = torch.from_numpy(base_feats).float().to(self.device)
        pooled_feats = self.multi_scale_pooling(base_tensor, pool_sizes=self.filter_sizes, modes=("avg", "max"))
        pooled_feats = pooled_feats.cpu().numpy()

        return np.concatenate([base_feats, pooled_feats], axis=1)
        

    def get_feature_tensor(self, images):
        if self.feature_extractor == "handcrafted":
            return self.get_handcrafted_features(images, n_jobs=self.n_cpus)
        raise ValueError(f"Unknown feature extractor {self.feature_extractor}")
    
    def get_dataloaders(self, data):
        return super().get_dataloaders(data)
    
    def train(self, model, epochs=10, lr=0.01, n_cpus=8):
        # CPU threading
        torch.set_num_threads(n_cpus)
        print(f"🔧 Using {torch.get_num_threads()} CPU threads")

        optimizer = torch.optim.Adam(model.params, lr=lr, weight_decay=1e-4)
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.params)}")

        # --- Prepare tensors ---
        feature_tensor = torch.from_numpy(self.feature_tensor).float().to(self.device)
        target = torch.from_numpy(self.train_masks).long().to(self.device)
        val_feature_tensor = (
            torch.from_numpy(self.val_feature_tensor).float().to(self.device)
            if self.val_feature_tensor is not None else None
        )
        val_target = (
            torch.from_numpy(self.val_masks).long().to(self.device)
            if self.val_masks is not None else None
        )

        # --- Handle extra dimensions ---
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if val_target is not None and val_target.dim() == 4 and val_target.size(1) == 1:
            val_target = val_target.squeeze(1)

        # --- Compute class balance ---
        flat = target.view(-1)
        counts = torch.bincount(flat, minlength=self.n_classes).float()
        counts[counts == 0] = 1.0
        balance = (1.0 / counts)
        balance = balance / balance.sum()
        print("Class balance weights:", balance.cpu().numpy())

        # --- Loss ---
        self.criterion = DiceCELoss(pos_weight=balance.detach().clone().to(self.device)).to(self.device)

        # --- Dataset and DataLoader ---
        #TODO
        train_loader = self.get_dataloaders(TensorDataset(feature_tensor, target))

        if val_feature_tensor is not None and val_target is not None:
            val_loader = self.get_dataloaders(TensorDataset(val_feature_tensor, val_target))
        else:
            val_loader = None

        # --- Logging containers ---
        train_losses, val_losses = [], []
        train_dices, val_dices = [], []
        train_ious, val_ious = [], []
        train_clious, val_clious = [], []

        # --- Training loop ---
        pbar = tqdm.tqdm(range(epochs), desc="Training PCNet")
        for epoch in pbar:
            batch_losses, batch_dices, batch_ious, batch_clious = [], [], [], []

            for ft_batch, t_batch in train_loader:
                optimizer.zero_grad()
                logits = model.evaluate(ft_batch)  # Ensure evaluate supports batched inputs
                loss = self.criterion(logits, t_batch)
                loss.backward()
                optimizer.step()
                # Convert logits to predicted binary mask (foreground class = 1)
                pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()  # [4, 1, 256, 256]
                true_mask = t_batch.unsqueeze(1).float()                       # [4, 1, 256, 256]

                # Metrics
                dice_score = self.dice_coef(logits, t_batch).item()
                iou_score = self.iou_score(logits, t_batch).item()

                cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

                batch_losses.append(loss.item())
                batch_dices.append(dice_score)
                batch_ious.append(iou_score)
                batch_clious.append(cl_iou_score)

            # --- Log training metrics ---
            epoch_loss = np.mean(batch_losses)
            epoch_dice = np.mean(batch_dices)
            epoch_iou = np.mean(batch_ious)
            epoch_cl_iou = np.mean(batch_clious)

            train_losses.append(epoch_loss)
            train_dices.append(epoch_dice)
            train_ious.append(epoch_iou)
            train_clious.append(epoch_cl_iou)

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_dice", epoch_dice, step=epoch)
            mlflow.log_metric("train_iou", epoch_iou, step=epoch)
            mlflow.log_metric("train_cl_iou", epoch_cl_iou, step=epoch)

            # --- Validation ---
            if val_loader is not None:
                val_loss, val_dice, val_iou, val_cl_iou = self.evaluate(model, val_loader)
                val_losses.append(val_loss)
                val_dices.append(val_dice)
                val_ious.append(val_iou)
                val_clious.append(val_cl_iou)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_dice", val_dice, step=epoch)
                mlflow.log_metric("val_iou", val_iou, step=epoch)
                mlflow.log_metric("val_cl_iou", val_cl_iou, step=epoch)

            pbar.set_postfix({
                "train_loss": f"{epoch_loss:.4f}",
                "val_loss": f"{val_losses[-1]:.4f}" if val_loader else None
            })

        # --- Save logs ---
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

        # --- Plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        if val_loader:
            plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title("Training Loss over Epochs")
        plt.savefig(os.path.join(self.log_path, "training_plot.png"))

        print("✅ Training complete.")
        with open(self.experiment_report, "a") as f:
            f.write(f"Final Training Loss: {train_losses[-1]}, Dice: {train_dices[-1]}, IoU: {train_ious[-1]}, CL IoU: {train_clious[-1]}\n")
            if val_loader:
                f.write(f"Final Validation Loss: {val_losses[-1]}, Dice: {val_dices[-1]}, IoU: {val_ious[-1]}, CL IoU: {val_clious[-1]}\n")
        f.close()
        return model

    # --- Parallelized Evaluation ---
    def evaluate(self, model, dataloader):
        losses, dices, ious, clious = [], [], [], []
        with torch.no_grad():
            for ft_batch, t_batch in dataloader:
                logits = model.evaluate(ft_batch)
                loss = self.criterion(logits, t_batch)
                dice_score = self.dice_coef(logits, t_batch).item()
                iou_score = self.iou_score(logits, t_batch).item()

                pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()  # [4, 1, 256, 256]
                true_mask = t_batch.unsqueeze(1).float()                       # [4, 1, 256, 256]
                cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

                losses.append(loss.item())
                dices.append(dice_score)
                ious.append(iou_score)
                clious.append(cl_iou_score)

        mean_loss, mean_dice, mean_iou, mean_cl_iou = np.mean(losses), np.mean(dices), np.mean(ious), np.mean(clious)
        
        return mean_loss, mean_dice, mean_iou, mean_cl_iou
    
    def show_prediction(self, model):
        with open(self.experiment_report, "a") as f:
            f.write(str(model) + "\n")
        f.close()
        predictions_path = os.path.join(self.log_path, "predictions")
        os.makedirs(predictions_path, exist_ok=True)
        imgs = [self.test_data[i][0] for i in range(len(self.test_data))]
        for enum, (img, inputs, gt) in enumerate(zip(imgs, self.test_inputs, self.test_ground_truth)):
            # probs = model.evaluate(img)
            mask = model.predict(inputs)
            print(mask.shape, mask.min(), mask.max(), mask.mean())
            
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            plt.figure(figsize=(12,6))
            plt.subplot(1,3,1)
            plt.imshow(img_np)
            plt.title("Input Image")
            plt.axis("off")
            plt.subplot(1,3,2)
            plt.imshow(gt_np, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")
            plt.subplot(1,3,3)
            plt.imshow(mask[0][0], cmap="cool")
            plt.title("Prediction")
            plt.axis("off")
            plt.savefig(os.path.join(predictions_path, f"prediction_{enum}.png")) 

        print_debugging(self.log_path)
        model.visualize(save_path=self.log_path)

    def finetune(self):
        pass

    def save_model(self, model_name="pcnet"):
        with open(os.path.join(self.save_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        f.close()

    def load_model(self, model_name="pcnet"):
        with open(os.path.join(self.save_path, f"{model_name}.pkl"), "rb") as f:
            self.model = pickle.load(f)
        f.close()
        
    def run(self):
        self.set_dataset(self.data_path, self.test_path)
        print_info("Dataset size:", len(self.data))
        if self.execution_mode == "train":
            self.set_algorithm()
            self.loss_function = BCEDiceLoss()

            self.feature_tensor = self.get_feature_tensor(np.array([self.train_data[i][0] for i in range(len(self.train_data))]))
            self.train_masks = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
            self.val_feature_tensor = self.get_feature_tensor(np.array([self.val_data[i][0] for i in range(len(self.val_data))]))
            self.val_masks = np.array([self.val_data[i][1] for i in range(len(self.val_data))]) if self.validation_split > 0 else None

            self.model.init_network(self.feature_tensor, self.train_masks)  
            self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            if self.test_path is not None:
                self.test_inputs = torch.from_numpy(self.get_feature_tensor(np.array([self.test_data[i][0] for i in range(len(self.test_data))]))).float().to(self.device)
                self.test_ground_truth = torch.from_numpy(np.array([self.test_data[i][1] for i in range(len(self.test_data))])).long().to(self.device)
                if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)
                test_loader = self.get_dataloaders(TensorDataset(self.test_inputs, self.test_ground_truth))
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

class ExperimentGEPC(ExperimentPC):
    def __init__(self, config):
        super().__init__(config)
        self.ge = GEOptimizer(
                    pop_size=2,
                    genome_len=96,
                    max_depth=self.max_depth,          # search budget
                    max_branching=self.max_branching,      # upper bound for sum fan-in
                    crossover_rate=0.9,
                    mutation_rate=0.05,
                    elite_k=2,
                    device="cpu",
                )
        
    def run(self):
        self.set_dataset(self.data_path, self.test_path)
        print_info("Dataset size:", len(self.data))
        if self.execution_mode == "train":
            self.set_algorithm()
            self.loss_function = BCEDiceLoss()

            self.feature_tensor = self.get_feature_tensor(np.array([self.train_data[i][0] for i in range(len(self.train_data))]))
            self.train_masks = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
            self.val_feature_tensor = self.get_feature_tensor(np.array([self.val_data[i][0] for i in range(len(self.val_data))]))
            self.val_masks = np.array([self.val_data[i][1] for i in range(len(self.val_data))]) if self.validation_split > 0 else None

            ge_train_size = int(0.1 * len(self.train_data))
            ge_X_train = torch.from_numpy(self.feature_tensor).float().to(self.device)[:ge_train_size]
            ge_Y_train = torch.from_numpy(self.train_masks).long().to(self.device)[:ge_train_size]
            best_pc, best_genome, history = self.ge.evolve(ge_X_train, ge_Y_train, self.epochs)
            print_info("Best genome:", best_genome)
            print_info("Best PC:", str(best_pc))
            self.model = best_pc

            self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            if self.test_path is not None:
                self.test_inputs = torch.from_numpy(self.get_feature_tensor(np.array([self.test_data[i][0] for i in range(len(self.test_data))]))).float().to(self.device)
                self.test_ground_truth = torch.from_numpy(np.array([self.test_data[i][1] for i in range(len(self.test_data))])).long().to(self.device)
                if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)
                test_loader = self.get_dataloaders(TensorDataset(self.test_inputs, self.test_ground_truth))
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