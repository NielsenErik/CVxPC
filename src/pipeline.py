
import mlflow

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import os
import glob
from networks import *

import tqdm

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

class CrackDataset(Dataset):
    '''
    Crack Dataset for image segmentation tasks.
    Args:
        images (list): List of image file paths or numpy arrays.
        ground_truth (list): List of corresponding ground truth masks.
        transform (callable, optional): Optional transform to be applied on a sample.
        resize_size (tuple, optional): Desired output size (width, height) for resizing images and masks.
    Returns:
        torch.utils.data.Dataset: Dataset object for loading crack images and masks.
    '''
    def __init__(self, images_path, ground_truths_path = None, resize_size=(256, 256), sample_size=None):
        self.sample_size = sample_size
        image_exts = ["*.jpg", "*.jpeg", "*.png"]
        mask_exts = ["*.jpg", "*.jpeg", "*.png"]
        if images_path.endswith('jpg') or images_path.endswith('jpeg') or images_path.endswith('png'):
            self.img_paths = [images_path]
            
        else:
            self.img_paths = sorted(
                    [p for ext in image_exts for p in glob.glob(os.path.join(images_path, ext))]
                )
        if sample_size is not None:
            self.img_paths = self.img_paths[:sample_size]
        self.resize_size = resize_size

    def resize_img(self, image, w, h):
        transform = transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor()
        ])
        return transform(image) 

    def resize_gt(self, gt, w, h):
        transform = transforms.Compose([
            transforms.Resize((w, h), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        return transform(gt)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")   # 3 channels
        # img = self.resize_img(img, self.resize_size[0], self.resize_size[1])
        img = transforms.Compose([
            transforms.Resize((self.resize_size[0], self.resize_size[1])),
            transforms.ToTensor()
        ])(img)
        return img
    

class Pipeline:
    def __init__(self, images_path, ground_truths_path = None, resize_size=(256, 256), sample_size=None):
        self.images_path = images_path
        self.ground_truths_path = ground_truths_path
        self.resize_size = resize_size
        self.sample_size = sample_size

    def get_dataset(self):
        return CrackDataset(self.images_path, self.ground_truths_path, self.resize_size, self.sample_size)

    def load_model(self, model_path="models/unet.pth"):
        params = torch.load(model_path, map_location=torch.device('cpu'))
        model = UNet(n_classes=1)
        model.load_state_dict(params)
        return model
    
    def log_params(self, params):
        mlflow.log_params(params)

    def predict(self, model, dataset):
        transform = transforms.Compose([
            transforms.ToPILImage(),    
            transforms.Resize((533, 800)),
            transforms.ToTensor()
        ])
        model.eval()
        predictions = []
        data = torch.utils.data.DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            for img in data:
                output = model(img)
                output= output.squeeze(0)   
                output = transform(output)
                predictions.append(output)

        return predictions
    
    def from_predictions_to_imgs(self, predictions):
        imgs = []
        for pred in predictions:
            pred_img = transforms.ToPILImage()(pred.squeeze(0))
            imgs.append(pred_img)
        return imgs

    def add_n_classes(self, model, additional_classes):
        n_classes = model.n_classes + additional_classes
        if isinstance(model, UNet, UNetPlusPlus):
            model.n_classes = n_classes
            model.conv_last = nn.Conv2d(64, n_classes, 1)
        else:
            raise NotImplementedError("add_n_classes is not implemented for this model type.")
        return model
    
    def finetune(self, model, imgs, masks, epochs=50, additional_classes=None):
        if additional_classes is not None:
            model = self.add_n_classes(model, additional_classes)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(imgs, masks), batch_size=16, shuffle=True)
        criterion = DiceCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Block all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the last convolution block(s)
        for param in model.dconv_up1.parameters():
            param.requires_grad = True
        for param in model.conv_last.parameters():
            param.requires_grad = True

        # Training loop (single epoch for demonstration)
        pbar = tqdm.tqdm(total=epochs)
        model.train()
        for epoch in range(epochs):
            loss_epoch = 0.0
            for images, masks in dataloader:
                preds = model(images)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += loss.item()
            pbar.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss_epoch/len(dataloader)}")
        return model

if __name__ == "__main__":
    model_path = "models/unet.pth"
    pipeline = Pipeline(images_path="data/A22/images/", ground_truths_path="data/A22/masks/", resize_size=(256, 256), sample_size=5)
    dataset = pipeline.get_dataset()
    dataset = dataset  # Use first 5 samples for demonstration
    model = pipeline.load_model(model_path=model_path)
    model = pipeline.finetune(model, dataset, dataset, epochs=50, additional_classes=1)
    predictions = pipeline.predict(model, dataset)
    imgs = pipeline.from_predictions_to_imgs(predictions)
    for img in imgs:
        img.show()