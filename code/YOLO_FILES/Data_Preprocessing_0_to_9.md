```python
import os

import zipfile

import h5py

import cv2

import numpy as np

import torch

from google.colab import drive

  

# 1. Mount Google Drive

drive.mount('/content/drive')

  

# 2. Extract dataset zip to local Colab NVMe memory

zip_path = "/content/drive/MyDrive/YOLO_Digits/dataset_root.zip"

extract_path = "/content/dataset_root"

  

if not os.path.exists(extract_path):

    print("📦 Unzipping 3GB Dataset to fast local Colab disk...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        zip_ref.extractall("/content/")

    print("✅ Unzipped successfully!")

  

# 3. Preprocessing & HDF5 Compilation Function for 0-9 + BG

def build_h5_dataset(base_path, h5_output_path):

    BLOCK_SIZE = 41

    C_CONSTANT = 15

  

    image_list = []

    target_list = []

  

    folders = [str(i) for i in range(10)] + ['bg']  # '0' through '9' + 'bg'

  

    print(f"\n🔨 Processing and Compiling {h5_output_path}...")

  

    for folder in folders:

        p = os.path.join(base_path, folder)

        if not os.path.exists(p):

            continue

  

        is_bg = (folder == 'bg')

        print(f"  ➜ Compiling class: '{folder}'")

  

        for f in os.listdir(p):

            if not f.lower().endswith(('.jpg', '.png', '.jpeg')):

                continue

  

            img_path = os.path.join(p, f)

            txt_path = os.path.join(p, f.rsplit('.', 1)[0] + ".txt")

  

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None: continue

            h_orig, w_orig = img.shape

  

            # Square Padding

            if h_orig != w_orig:

                max_dim = max(h_orig, w_orig)

                pad_h, pad_w = (max_dim - h_orig) // 2, (max_dim - w_orig) // 2

                img_padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

            else:

                img_padded = img

                pad_h, pad_w, max_dim = 0, 0, w_orig

  

            # Adaptive Thresholding

            adaptive_mask = cv2.adaptiveThreshold(

                img_padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

                cv2.THRESH_BINARY_INV, BLOCK_SIZE, C_CONSTANT

            )

  

            resized_mask = cv2.resize(adaptive_mask, (448, 448))

  

            # Parse bounding boxes

            base_boxes = []

            if not is_bg and os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:

                with open(txt_path, 'r') as file_data:

                    for line in file_data.readlines():

                        parts = line.split()

                        if len(parts) < 5: continue

                        c, x, y, w, h = map(float, parts)

                        x_pad = ((x * w_orig) + pad_w) / max_dim

                        y_pad = ((y * h_orig) + pad_h) / max_dim

                        w_pad = (w * w_orig) / max_dim

                        h_pad = (h * h_orig) / max_dim

                        base_boxes.append([int(c), x_pad, y_pad, w_pad, h_pad])

  

            # Grid target construction: Shape [7, 7, 15] -> (1 Conf + 4 BBox + 10 One-Hot Classes for 0-9)

            target = np.zeros((7, 7, 15), dtype=np.float32)

            for box in base_boxes:

                c, x_new, y_new, w_new, h_new = box

                gx, gy = min(int(x_new * 7), 6), min(int(y_new * 7), 6)

                target[gy, gx, 0] = 1.0  # Confidence

                target[gy, gx, 1:5] = [x_new * 7 - gx, y_new * 7 - gy, w_new, h_new]  # BBox coords

                if 0 <= c <= 9:

                    target[gy, gx, 5 + c] = 1.0  # One-hot encoded class 0 to 9

  

            image_list.append(resized_mask)

            target_list.append(target)

  

    # Save to HDF5

    print("💾 Saving to HDF5 binary archive...")

    with h5py.File(h5_output_path, "w") as hf:

        hf.create_dataset("images", data=np.array(image_list, dtype=np.uint8), compression="gzip")

        hf.create_dataset("targets", data=np.array(target_list, dtype=np.float32), compression="gzip")

    print(f"✅ Created {h5_output_path} successfully!")

  

# Run compilation

build_h5_dataset("/content/dataset_root/train_set", "/content/yolo_train.h5")

build_h5_dataset("/content/dataset_root/test_set", "/content/yolo_test.h5")

  

# Copy compiled .h5 files to Drive for permanent safety

os.makedirs("/content/drive/MyDrive/YOLO_Digits/compiled", exist_ok=True)

!cp /content/yolo_train.h5 /content/drive/MyDrive/YOLO_Digits/compiled/

!cp /content/yolo_test.h5 /content/drive/MyDrive/YOLO_Digits/compiled/

print("☁️ Saved compiled .h5 files to Google Drive!")
```