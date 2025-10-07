from datetime import datetime
import string
import numpy as np

def get_logdir_name():
    """
    Returns a name for the dir
    :returns: a name in the format dd-mm-yyyy_:mm:ss_<random string>
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")
    rand_str = "".join(np.random.choice([*string.ascii_lowercase], 8))
    return f"{time}_{rand_str}"

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ORANGE = '\033[33m'
    RESULT =  '\033[94m'
    WARNING = '\033[93m'
    DEBUGGING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def now():
    return datetime.now().strftime("%H:%M:%S")

def print_error(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.FAIL+"ERROR"+colors.ENDC+"] " +" ".join(map(str,args))+"\n", **kwargs)
    
def print_warning(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.ORANGE+"WARNING"+colors.ENDC+"] " +" ".join(map(str,args))+ "\n", **kwargs)

def print_debugging(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.DEBUGGING+"DEBUG"+colors.ENDC+"] " +" ".join(map(str,args))+"\n", **kwargs)
    
def print_info(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"INFO"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)

def print_configs(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"CONFIGS"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)
    
def print_results(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.RESULT+"RESULTS"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)


def crackseg9k_split():
    import os
    import shutil

    # Base folder
    base_dir = "data/Crackseg9k"

    # Input folders
    images_dir = os.path.join(base_dir, "Images")
    masks_dir = os.path.join(base_dir, "Final_Masks", "Masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print("❌ Input directories do not exist. Please check the paths.")
        return
    # Output folders
    output_train_img = os.path.join(base_dir, "train", "images")
    output_train_mask = os.path.join(base_dir, "train", "masks")
    output_test_img = os.path.join(base_dir, "test", "images")
    output_test_mask = os.path.join(base_dir, "test", "masks")

    # Create output directories
    for path in [output_train_img, output_train_mask, output_test_img, output_test_mask]:
        os.makedirs(path, exist_ok=True)

    # Function to copy images/masks according to a txt file
    def copy_files(split):
        txt_path = os.path.join(base_dir, "Final_Masks", f"{split}.txt")
        with open(txt_path, "r") as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]

        print(f"📦 Processing {split}: {len(filenames)} files")

        for name in filenames:
            # Common pattern: image file might have .jpg/.png extension
            possible_exts = [".png"]
            image_file = None

            for ext in possible_exts:
                candidate = os.path.join(images_dir, name)

                if os.path.exists(candidate):
                    image_file = candidate
                    break

            mask_file = os.path.join(masks_dir, name)  # adjust if mask ext differs

            if image_file and os.path.exists(mask_file):
                if split == "train":
                    shutil.copy2(image_file, output_train_img)
                    shutil.copy2(mask_file, output_train_mask)
                elif split == "test":
                    shutil.copy2(image_file, output_test_img)
                    shutil.copy2(mask_file, output_test_mask)
            else:
                print(f"⚠️ Missing pair for: {name}")

    # Copy train/test sets
    copy_files("train")
    copy_files("test")

    print("✅ Dataset reorganized successfully!")