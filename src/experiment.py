import numpy as np
import cv2
import os
import mlflow
from networks import *
from probabilistic_circuits.probabilistic_circuits import *
import abc
from torch.utils.data import DataLoader
from tqdm import trange
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm


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

    else:
        raise ValueError(f"Algorithm {config['algorithm']} not supported")
    
class Experiment:
    def __init__(self, config):
        self.config = config
        self.execution_mode = self.config.get("execution_mode", "train")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_path = self.config.get("data_path", "data/train")
        self.test_path = self.config.get("test_path", "data/test")
        self.save_path = self.config.get("save_path", "models/")

        self.batch_size = self.config.get("batch_size", 16)
        self.sample_size = self.config.get("sample_size", 20)
        self.resize_size = self.config.get("resize_size", [256, 256])
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.threshold = self.config.get("threshold", 0.5)
        
        self.epochs = self.config.get("epochs", 10)
        self.model = None

    
    @abc.abstractmethod
    def set_algorithm(self):
        pass

    def set_dataset(self,train_path, test_path=None):
        self.images_path = train_path + "/images/"
        self.ground_truth_path = train_path + "/masks/"
        if test_path is not None:
            self.test_images_path = test_path + "/images/"
            self.test_ground_truth_path = test_path + "/masks/"
            self.test_data = CrackDataset(self.test_images_path, self.test_ground_truth_path, augmentation_percentage=0.0,sample_size=10, resize_size=(self.resize_size[0], self.resize_size[1]))
        self.data = CrackDataset(self.images_path, self.ground_truth_path, augmentation_percentage=0.1,sample_size=self.sample_size, resize_size=(self.resize_size[0], self.resize_size[1]))     

    @abc.abstractmethod
    def setup(self, pos_weight=1.0):
        pass

    @abc.abstractmethod
    def get_dataloaders(self, data):
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

    def get_dataloaders(self, data):
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
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
    # Metrics
    # ----------------------------
    def dice_coef(self, pred, target, smooth=1.0):
        pred = (torch.sigmoid(pred) > 0.5).float()
        intersection = (pred * target).sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
        return dice.mean().item()

    def iou_score(self, pred, target, smooth=1.0):
        pred = (torch.sigmoid(pred) > 0.5).float()
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean().item()


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

        for epoch in trange(epochs):
            model.train()
            epoch_loss, epoch_dice, epoch_iou = 0, 0, 0

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

            train_loss = epoch_loss / len(train_loader)
            train_dice = epoch_dice / len(train_loader)
            train_iou  = epoch_iou / len(train_loader)

            # Validation
            if val_loader:
                model.eval()
                val_loss, val_dice, val_iou = 0, 0, 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        val_loss += loss_function(outputs, masks).item()
                        val_dice += self.dice_coef(outputs, masks)
                        val_iou  += self.iou_score(outputs, masks)

                val_loss /= len(val_loader)
                val_dice /= len(val_loader)
                val_iou  /= len(val_loader)
            else:
                val_loss, val_dice, val_iou = None, None, None

            if scheduler:
                scheduler.step()

            # Logging
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f} | "
                f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}" if val_loader else "")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_dice", train_dice, step=epoch)
            mlflow.log_metric("train_iou", train_iou, step=epoch)
            if val_loader:
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_dice", val_dice, step=epoch)
                mlflow.log_metric("val_iou", val_iou, step=epoch)

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
        return test_loss/len(test_loader)        

    def log_results(self):
        pass

    def show_predictions(self, model, dataset, device, num_samples=3, threshold=0.5):
        model.eval()
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
            split_idx = int(0.8 * len(self.data))
            train_data = torch.utils.data.Subset(self.data, list(range(0, split_idx)))
            val_data = torch.utils.data.Subset(self.data, list(range(split_idx, len(self.data))))
            self.model = self.train(self.model, train_data=train_data, val_data=val_data, loss_function=self.loss, optimizer=self.optimizer, device=self.device, epochs=self.epochs, batch_size=self.batch_size)
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
        self.input_nodes = []

    def tensor_to_features(self, image_tensor):
        """
        Convert tensor image to feature matrix with RGB + XY.
        
        :param image_tensor: shape (C,H,W) or (H,W,C), tensor or numpy
        :return: np.ndarray of shape (H*W, C+2)
        """
        # Convert to numpy if tensor
        if not isinstance(image_tensor, np.ndarray):
            image = image_tensor.cpu().numpy()
        else:
            image = image_tensor

        # Handle channel-first
        if image.ndim == 3 and image.shape[0] <= 4:  # (C,H,W)
            image = np.transpose(image, (1, 2, 0))   # -> (H,W,C)

        H, W, C = image.shape

        # Flatten RGB
        pixels = image.reshape(-1, C)

        # XY coords normalized
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coords = np.stack([xx.flatten() / W, yy.flatten() / H], axis=1)

        # Final features = [R,G,B,x,y]
        features = np.concatenate([pixels, coords], axis=1).astype(np.float32)
        return features
    def set_input_nodes(self, input_images):
        input_nodes = []
        features = self.tensor_to_features(input_images[0])
        n_features = features.shape[1]
        print("Number of features:", n_features)
        for feat_idx in range(n_features):
            # Start with a generic Gaussian PDF
            def distribution(v, mu=0.0, sigma=1.0):
                return norm.pdf(v, loc=mu, scale=sigma+1e-12)
            node = InputNode(feature_idx=feat_idx, distribution=distribution)
            input_nodes.append(node)
        return input_nodes

    def set_algorithm(self):
        self.model = PCNet(n_classes=1, distribution=None, device="cpu", max_depth=4, max_branching=3)

    def train(self, model):
        X = np.array([self.data[i][0] for i in range(len(self.data))])
        y = np.array([self.data[i][1] for i in range(len(self.data))])
        model.fit_supervised(X, y, epochs=self.epochs)
        return model
    
    def evaluate(self, model, data, loss_function):
        for d in data:            
            probs= model.predict(d[0])
            print(probs.shape, probs.min(), probs.max(), probs.mean())
            plt.imshow(probs)
            plt.show()
    
    def show_prediction(self, model, image_list, ground_truth_list):

        for img, gt in zip(image_list, ground_truth_list):
            # probs = model.evaluate(img)
            mask = model.predict_mask(img)
            probs = torch.sigmoid(torch.tensor(mask)).numpy()
            print(mask.shape, mask.min(), mask.max(), mask.mean())
            class_1 = probs[0][0]
            class_2 = probs[0][1]
            print("Class 1:", class_1.min(), class_1.max(), class_1.mean())
            print("Class 2:", class_2.min(), class_2.max(), class_2.mean())

            print(probs.shape, probs.min(), probs.max(), probs.mean())
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()
            plt.figure(figsize=(12,6))
            plt.subplot(1,4,1)
            plt.imshow(img_np)
            plt.title("Input Image")
            plt.axis("off")
            plt.subplot(1,4,2)
            plt.imshow(gt_np, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")
            plt.subplot(1,4,3)
            plt.imshow(probs[0][0], cmap="cool")
            plt.title("Prediction")
            plt.axis("off")
            plt.subplot(1,4,4)
            plt.imshow(probs[0][1], cmap="cool")
            plt.title("Prediction")
            plt.axis("off")
            plt.show()

        model.visualize()

    def finetune(self):
        pass
        
    def run(self):
        self.set_dataset(self.data_path, self.test_path)
        print("Dataset size:", len(self.data))
        self.set_algorithm()
        self.loss_function = BCEDiceLoss()
        self.model = self.train(self.model)
        if self.test_path is not None:
            images = [self.test_data[i][0] for i in range(len(self.test_data))]
            gts = [self.test_data[i][1] for i in range(len(self.test_data))]
            self.show_prediction(self.model, images, gts)

        
