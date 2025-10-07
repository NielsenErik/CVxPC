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
    def __init__(self, images_path, ground_truths_path, resize_size=(256, 256), augmentation_percentage=None, sample_size=224):
        self.sample_size = sample_size
        image_exts = ["*.jpg", "*.jpeg", "*.png"]
        mask_exts = ["*.jpg", "*.jpeg", "*.png"]

        self.img_paths = sorted(
            [p for ext in image_exts for p in glob.glob(os.path.join(images_path, ext))]
        )[:sample_size]
        self.mask_paths = sorted(
            [p for ext in mask_exts for p in glob.glob(os.path.join(ground_truths_path, ext))]
        )[:sample_size]
        self.resize_size = resize_size
        self.augmentation_percentage = augmentation_percentage
        

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
        mask = Image.open(self.mask_paths[idx]).convert("L")   # 1 channel (grayscale)
        
        img = self.resize_img(img, self.resize_size[0], self.resize_size[1])
        mask = self.resize_gt(mask, self.resize_size[0], self.resize_size[1])

        return img, mask
    
if __name__ == "__main__":
    pass