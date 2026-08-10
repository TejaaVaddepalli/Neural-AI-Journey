```python
import os

import h5py

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np

import random

import cv2

from google.colab import drive

  

# Ensure Drive is mounted

if not os.path.exists("/content/drive/MyDrive"):

    drive.mount('/content/drive')

  

# Ensure save directory exists

save_dir = "/content/drive/MyDrive/YOLO_Digits"

os.makedirs(save_dir, exist_ok=True)

best_model_path = os.path.join(save_dir, "best_yolo_0to9_model.pth")

  

# =====================================================================

# 1. DYNAMIC ON-THE-FLY AUGMENTATION DATASET (0-9 DIGITS)

# =====================================================================

class OnTheFlyYOLODataset(Dataset):

    def __init__(self, h5_file_path, augment=True):

        with h5py.File(h5_file_path, "r") as hf:

            self.images = hf["images"][:]

            self.targets = hf["targets"][:]

        self.augment = augment

  

    def __len__(self):

        return len(self.images)

  

    def __getitem__(self, idx):

        mask = self.images[idx].copy()

        target = self.targets[idx].copy()

  

        # Dynamic Augmentation loop during training

        if self.augment and random.random() > 0.3:

            h_m, w_m = mask.shape

            angle = random.uniform(-15, 15)

            shift_x = random.uniform(-0.08, 0.08) * w_m

            shift_y = random.uniform(-0.08, 0.08) * h_m

  

            # Rotate & Translate Image

            M_rot = cv2.getRotationMatrix2D((w_m // 2, h_m // 2), angle, 1.0)

            mask = cv2.warpAffine(mask, M_rot, (w_m, h_m), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

  

            M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

            mask = cv2.warpAffine(mask, M_trans, (w_m, h_m), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

  

            # Re-adjust bounding box centers in grid

            new_target = np.zeros_like(target)

            for gy in range(7):

                for gx in range(7):

                    if target[gy, gx, 0] == 1.0:  # Object present

                        dx, dy, w, h = target[gy, gx, 1:5]

                        classes = target[gy, gx, 5:]

  

                        cx_pix = ((gx + dx) / 7.0 * w_m) + shift_x

                        cy_pix = ((gy + dy) / 7.0 * h_m) + shift_y

  

                        pts = np.dot(M_rot, np.array([[cx_pix], [cy_pix], [1]]))

                        new_x = np.clip(pts[0][0] / w_m, 0.0, 1.0)

                        new_y = np.clip(pts[1][0] / h_m, 0.0, 1.0)

  

                        new_gx, new_gy = min(int(new_x * 7), 6), min(int(new_y * 7), 6)

                        new_target[new_gy, new_gx, 0] = 1.0

                        new_target[new_gy, new_gx, 1:5] = [new_x * 7 - new_gx, new_y * 7 - new_gy, w, h]

                        new_target[new_gy, new_gx, 5:] = classes

            target = new_target

  

        img_tensor = torch.from_numpy(mask / 255.0).float().unsqueeze(0)

        target_tensor = torch.from_numpy(target).float()

        return img_tensor, target_tensor

  

# =====================================================================

# 2. PURE YOLO ARCHITECTURE FOR 0-9 DIGITS (15 CHANNELS)

# =====================================================================

class PureYOLONet0to9(nn.Module):

    def __init__(self):

        super().__init__()

        def conv_block(in_f, out_f):

            return nn.Sequential(

                nn.Conv2d(in_f, out_f, 3, padding=1),

                nn.BatchNorm2d(out_f),

                nn.LeakyReLU(0.1),

                nn.MaxPool2d(2, 2)

            )

        self.features = nn.Sequential(

            conv_block(1, 16), conv_block(16, 32), conv_block(32, 64),

            conv_block(64, 128), conv_block(128, 256), conv_block(256, 512)

        )

        # 15 output channels = 1 (Conf) + 4 (Coords) + 10 (Classes 0-9)

        self.yolo_head = nn.Sequential(nn.Conv2d(512, 15, 1), nn.Sigmoid())

  

    def forward(self, x):

        x = self.yolo_head(self.features(x))

        x = x.permute(0, 2, 3, 1)  # Shape: [Batch, 7, 7, 15]

        return x

  

# =====================================================================

# 3. METRIC EVALUATION FUNCTION

# =====================================================================

def compute_yolo_loss(out, tgs, criterion, W_COORD=30.0, W_OBJ=25.0, W_NOOBJ=1.5, W_CLASS=10.0):

    mask_obj = tgs[:, :, :, 0].unsqueeze(-1)

    mask_noobj = 1 - mask_obj

  

    loss_coord = criterion(out[:, :, :, 1:5] * mask_obj, tgs[:, :, :, 1:5] * mask_obj) * W_COORD

    loss_obj = criterion(out[:, :, :, 0] * mask_obj.squeeze(-1), tgs[:, :, :, 0] * mask_obj.squeeze(-1)) * W_OBJ

    loss_noobj = criterion(out[:, :, :, 0] * mask_noobj.squeeze(-1), tgs[:, :, :, 0] * mask_noobj.squeeze(-1)) * W_NOOBJ

    loss_class = criterion(out[:, :, :, 5:] * mask_obj, tgs[:, :, :, 5:] * mask_obj) * W_CLASS

  

    return loss_coord + loss_obj + loss_noobj + loss_class

  

def calculate_accuracy(pred, target):

    obj_mask = (target[:, :, :, 0] == 1.0)

    if obj_mask.sum() == 0:

        return 100.0, 100.0

  

    # Object Detection Confidence Accuracy (Threshold @ 0.4)

    pred_conf = (pred[:, :, :, 0] > 0.4)

    det_correct = (pred_conf[obj_mask] == True).float().sum()

    det_acc = (det_correct / obj_mask.sum()) * 100.0

  

    # Digit Classification Accuracy (0-9)

    pred_classes = torch.argmax(pred[:, :, :, 5:], dim=-1)

    true_classes = torch.argmax(target[:, :, :, 5:], dim=-1)

    class_correct = (pred_classes[obj_mask] == true_classes[obj_mask]).float().sum()

    class_acc = (class_correct / obj_mask.sum()) * 100.0

  

    return det_acc.item(), class_acc.item()

  

# =====================================================================

# 4. TRAINING & VALIDATION PIPELINE

# =====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  

train_dataset = OnTheFlyYOLODataset("/content/yolo_train.h5", augment=True)

test_dataset = OnTheFlyYOLODataset("/content/yolo_test.h5", augment=False)

  

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  

model = PureYOLONet0to9().to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

criterion = nn.MSELoss()

  

best_val_loss = float("inf")

EPOCHS = 100

  

print(f"\n🚀 HARDWARE ONLINE: {device} | Train Batches: {len(train_loader)} | Test Samples: {len(test_dataset)}")

print("=" * 75)

  

for epoch in range(EPOCHS):

    # --- TRAINING PHASE ---

    model.train()

    train_loss = 0.0

  

    for imgs, tgs in train_loader:

        imgs, tgs = imgs.to(device), tgs.to(device)

        optimizer.zero_grad()

        out = model(imgs)

  

        loss = compute_yolo_loss(out, tgs, criterion)

        loss.backward()

  

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

  

    scheduler.step()

    avg_train_loss = train_loss / len(train_loader)

  

    # --- VALIDATION PHASE ---

    model.eval()

    val_loss = 0.0

    val_det_acc, val_cls_acc = 0.0, 0.0

  

    with torch.no_grad():

        for imgs, tgs in test_loader:

            imgs, tgs = imgs.to(device), tgs.to(device)

            out = model(imgs)

  

            loss = compute_yolo_loss(out, tgs, criterion)

            val_loss += loss.item()

  

            d_acc, c_acc = calculate_accuracy(out, tgs)

            val_det_acc += d_acc

            val_cls_acc += c_acc

  

    avg_val_loss = val_loss / len(test_loader)

    avg_det_acc = val_det_acc / len(test_loader)

    avg_cls_acc = val_cls_acc / len(test_loader)

  

    # --- CHECKPOINT SAVING (BEST MODEL ONLY) ---

    is_best = avg_val_loss < best_val_loss

    if is_best:

        best_val_loss = avg_val_loss

        checkpoint = {

            "epoch": epoch + 1,

            "model_state": model.state_dict(),

            "optimizer_state": optimizer.state_dict(),

            "val_loss": best_val_loss,

            "det_acc": avg_det_acc,

            "cls_acc": avg_cls_acc

        }

        torch.save(checkpoint, best_model_path)

  

    # --- LOGGING ---

    saved_flag = "  [💾 NEW BEST SAVED!]" if is_best else ""

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Det Acc: {avg_det_acc:.1f}% | Digit Acc: {avg_cls_acc:.1f}%{saved_flag}")

  

print("=" * 75)

print(f"✅ Training Complete! Best model saved to: {best_model_path}")

print(f"🏆 Lowest Val Loss Achieved: {best_val_loss:.4f}")
```