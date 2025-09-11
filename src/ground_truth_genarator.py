import json
import cv2
import numpy as np

def from_json_to_mask(json_path, output_mask_path, output_overlay_path, image_path=None):
    """
    Convert Label Studio polygon annotations into a binary mask and overlay.

    Args:
        json_path (str): Path to the Label Studio JSON export file.
        output_mask_path (str): Path where the binary mask will be saved.
        output_overlay_path (str): Path where the overlay visualization will be saved.
        image_path (str, optional): Path to the original image (required for overlay).
    """
    # Load Label Studio JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Assume one annotation JSON entry
    ann = data[0]

    # Extract original size
    width = ann["OriginalWidth"]
    height = ann["OriginalHeight"]

    # Parse the "Value" string to JSON
    value = json.loads(ann["Value"])

    # Convert percentage coords → pixel coords
    points = value["points"]  # [[x%, y%], ...]
    polygon = np.array(
        [[int(x / 100 * width), int(y / 100 * height)] for x, y in points],
        dtype=np.int32
    )

    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Save mask
    cv2.imwrite(output_mask_path, mask)

    # If original image is provided → overlay
    if image_path is not None:
        image = cv2.imread(image_path)
        overlay = image.copy()
        overlay[mask == 255] = (0, 0, 255)  # red
        cv2.imwrite(output_overlay_path, overlay)
        print(f"Overlay saved to {output_overlay_path}")

    print(f"Mask saved to {output_mask_path}")

