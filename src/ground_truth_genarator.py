import json
import cv2
import numpy as np

# For removing BG

from PIL import Image
from skimage import io
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import os
import pandas as pd
from torchvision import transforms

def from_csv_to_masks(csv_path, image_dir, output_mask_dir, output_overlay_dir):
    """
    Convert polygon annotations from a CSV (like Label Studio export) into binary masks and overlays.

    Args:
        csv_path (str): Path to the CSV file containing annotations.
        image_dir (str): Directory containing original images.
        output_mask_dir (str): Directory to save mask images.
        output_overlay_dir (str): Directory to save overlay visualizations.
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_overlay_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        file_name = row["id"] + "_no_bg.png"  # Assuming images are PNGs
        annotations = row["annotations"]

        if pd.isna(annotations):
            print(f"Skipping {file_name}: no annotations")
            continue

        # Try to parse the list of annotation dicts
        try:
            ann_list = json.loads(annotations)
        except json.JSONDecodeError:
            try:
                # Sometimes single quotes or double-encoded JSON appear
                ann_list = json.loads(annotations.replace("'", "\""))
            except Exception:
                print(f"Invalid annotation format for {file_name}")
                continue

        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in ann_list:
            if ann.get("Type") != "polygonlabels":
                continue

            # Nested JSON string
            try:
                value = json.loads(ann["Value"])
            except Exception as e:
                print(f"Failed to parse Value for {file_name}: {e}")
                continue

            points = value.get("points", [])
            labels = value.get("polygonlabels", [])
            label_name = labels[0] if labels else "Unknown"

            if not points:
                continue

            # Scale to pixel coordinates
            polygon = np.array(
                [[int(x / 100 * width), int(y / 100 * height)] for x, y in points],
                dtype=np.int32
            )
            cv2.fillPoly(mask, [polygon], 255)

        # Save binary mask
        mask_path = os.path.join(output_mask_dir, f"{os.path.splitext(file_name)[0]}_mask.png")
        cv2.imwrite(mask_path, mask)

        # Create overlay for visualization
        overlay = image.copy()
        overlay[mask == 255] = (0, 0, 255)
        overlay_path = os.path.join(output_overlay_dir, f"{os.path.splitext(file_name)[0]}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        print(f"Saved mask and overlay for {file_name}")


def background_removal(image_dir, output_path):
    
    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
    os.makedirs(output_path, exist_ok=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cpu')
    birefnet.eval()
    birefnet.half()

    def extract_object(birefnet, imagepath):
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(imagepath)
        input_images = transform_image(image).unsqueeze(0).to('cpu').half()

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image, mask

    images_files = os.listdir(image_dir)
    images_path = [os.path.join(image_dir, img) for img in images_files]

    for image_path in images_path:
        orig_im = io.imread(image_path)
        orig_im_size = orig_im.shape[0:2]
        model_input_size = [1024, 1024]

        # process image
        pil_image = Image.open(image_path).convert("RGB")
        result_image = extract_object(birefnet, image_path)[0]

        # apply alpha mask
        pil_mask_im = Image.fromarray(result_image)
        orig_image = Image.open(image_path).convert("RGBA")
        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(pil_mask_im)

        # build output filename
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(output_path, f"{name}_no_bg.png")

        # save result
        no_bg_image.save(output_file)
        print(f"Saved: {output_file}")
        

if __name__ == "__main__":
    # Example usage:
    # from_json_to_mask("path/to/label_studio_export.json", "output/mask.png", "output/overlay.png", "path/to/original_image.png")
    output_dir = "data/A22/no_bg/images"
    background_removal("data/A22/images", output_dir)
    print(f"Background removed images are saved in {output_dir}")
    from_csv_to_masks("data/A22/stored_images.csv", "data/A22/no_bg/images", "data/A22/no_bg/masks", "data/A22/no_bg/overlays")