import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import mlflow
from PIL import Image
import glob
import matplotlib.pyplot as plt


class CrackAugmentations:
    def __init__(self):
        pass

    def random_augmentation(self, image):
        augmentations = [
            self.blur,
            self.rotate,
            self.flip,
            self.random_crop,
            self.to_grayscale,
            self.to_rgb
        ]
        return augmentations[np.random.randint(0, len(augmentations))](image)


    def blur(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def rotate(self, image, angle=90):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((angle, angle)),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def flip(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def random_crop(self, image, crop_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size),
            transforms.ToPILImage()
        ])
        return transform(image)
        
    def to_grayscale(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def to_rgb(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToPILImage()
        ])
        return transform(image)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, image):
        return self.transform(image)

class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, resize_size=(128, 128), sample_size=None, **kwargs):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

        # Collect all images and masks (any jpg/png/jpeg)
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.[pjPJ][pnPN][gG]")))
        mask_files  = sorted(glob.glob(os.path.join(masks_dir,  "*.[pjPJ][pnPN][gG]")))

        # Build mapping base_name -> full path (for all extensions)
        image_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        mask_map  = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

        # Keep only common bases
        common_bases = sorted(set(image_map.keys()) & set(mask_map.keys()))

        self.img_paths = [image_map[b] for b in common_bases]
        self.mask_paths  = [mask_map[b] for b in common_bases]

        if sample_size:
            self.img_paths = self.img_paths[:sample_size]
            self.mask_paths  = self.mask_paths[:sample_size]

        print(f"[INFO] CrackDataset paired: {len(self.img_paths)} image–mask pairs found "
              f"(from {len(image_files)} images, {len(mask_files)} masks)")

        self.resize_size = resize_size
        # add any transforms or augmentation setup here

    def __len__(self):
        return len(self.img_paths)


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
        if self.mask_paths is None:
            mask = None
        else:
            mask = Image.open(self.mask_paths[idx]).convert("L")   # 1 channel (grayscale)
        
        img = self.resize_img(img, self.resize_size[0], self.resize_size[1])
        if mask is not None:
            mask = self.resize_gt(mask, self.resize_size[0], self.resize_size[1])
            mask = (mask > 0.1).float()
        return img, mask
    
if __name__ == "__main__":
    pass