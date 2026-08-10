```python
import cv2

import torch

import torch.nn as nn

import numpy as np

import pickle

import os

  

# =====================================================================

# 1. DEFINE ARCHITECTURE FOR 0-9 DIGITS (15 CHANNELS)

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

        # 15 channels = 1 (Conf) + 4 (Coords) + 10 (Classes 0-9)

        self.yolo_head = nn.Sequential(nn.Conv2d(512, 15, 1), nn.Sigmoid())

    def forward(self, x):

        x = self.yolo_head(self.features(x))

        x = x.permute(0, 2, 3, 1)  # Output layout: [Batch, 7, 7, 15]

        return x

  

# =====================================================================

# 2. INITIALIZE MODEL & LOAD WEIGHTS

# =====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PureYOLONet0to9().to(device)

  

model_pkl_path = "best_yolo_0to9_model.pkl"

model_pth_path = "best_yolo_0to9_model.pth"

  

# Load from PKL if available, otherwise fallback to PTH

if os.path.exists(model_pkl_path):

    print(f"📦 Loading weights from PKL file: {model_pkl_path}")

    with open(model_pkl_path, "rb") as f:

        payload = pickle.load(f)

    model.load_state_dict(payload["model_state_dict"])

elif os.path.exists(model_pth_path):

    print(f"📦 Loading weights from PTH file: {model_pth_path}")

    checkpoint = torch.load(model_pth_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:

        model.load_state_dict(checkpoint["model_state"])

    else:

        model.load_state_dict(checkpoint)

else:

    raise FileNotFoundError("❌ Neither 'best_yolo_0to9_model.pkl' nor 'best_yolo_0to9_model.pth' was found!")

  

model.eval()

  

# Color palette for 10 distinct classes (BGR format)

CLASS_COLORS = [

    (0, 255, 255),   # 0: Yellow

    (255, 0, 0),     # 1: Blue

    (0, 255, 0),     # 2: Green

    (0, 165, 255),   # 3: Orange

    (255, 0, 255),   # 4: Magenta

    (255, 255, 0),   # 5: Cyan

    (128, 0, 128),   # 6: Purple

    (0, 128, 255),   # 7: Light Orange

    (0, 0, 255),     # 8: Red

    (0, 255, 128)    # 9: Spring Green

]

  

# =====================================================================

# 3. LIVE WEBCAM PIPELINE

# =====================================================================

cap = cv2.VideoCapture(0)

  

BLOCK_SIZE = 41

C_CONSTANT = 15

CONF_THRESHOLD = 0.35  # Confidence cutoff for live tracking

  

print("\n🚀 Live 0-9 Digit YOLO Tracker Active! Press 'q' to quit.")

  

while True:

    ret, frame = cap.read()

    if not ret:

        break

  

    display_frame = frame.copy()

    h_orig, w_orig = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  

    # 1. Square Padding Pipeline

    if h_orig != w_orig:

        max_dim = max(h_orig, w_orig)

        pad_h = (max_dim - h_orig) // 2

        pad_w = (max_dim - w_orig) // 2

        img_padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

    else:

        img_padded = gray

        pad_h, pad_w, max_dim = 0, 0, w_orig

  

    # 2. Adaptive Binarization Filter (Matching Dataset Preprocessing)

    adaptive_mask = cv2.adaptiveThreshold(

        img_padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV, BLOCK_SIZE, C_CONSTANT

    )

  

    # 3. Interpolate into 448x448 Neural Input Tensor

    resized_mask = cv2.resize(adaptive_mask, (448, 448))

    img_tensor = torch.from_numpy(resized_mask / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)

  

    # 4. Forward Pass Through Model

    with torch.no_grad():

        prediction = model(img_tensor).squeeze(0).cpu().numpy()  # Layout: [7, 7, 15]

  

    conf_grid = prediction[:, :, 0]

    gy, gx = np.unravel_index(np.argmax(conf_grid), conf_grid.shape)

    max_conf = conf_grid[gy, gx]

  

    # 5. Coordinate Un-padding & Visual Overlay

    if max_conf > CONF_THRESHOLD:

        tx, ty, tw, th = prediction[gy, gx, 1:5]

        class_scores = prediction[gy, gx, 5:]  # 10 output class probabilities

        predicted_digit = np.argmax(class_scores)

  

        # Scale cell offsets to relative coordinates (0.0 - 1.0)

        x_p = (tx + gx) / 7.0

        y_p = (ty + gy) / 7.0

  

        # Scale canvas coordinates to source pixel dimensions

        cx_scaled = (x_p * max_dim) - pad_w

        cy_scaled = (y_p * max_dim) - pad_h

        w_scaled = tw * max_dim

        h_scaled = th * max_dim

  

        # Calculate bounding box bounds

        x1 = max(int(cx_scaled - (w_scaled / 2)), 0)

        y1 = max(int(cy_scaled - (h_scaled / 2)), 0)

        x2 = min(int(cx_scaled + (w_scaled / 2)), w_orig)

        y2 = min(int(cy_scaled + (h_scaled / 2)), h_orig)

  

        # UI Styling

        box_color = CLASS_COLORS[predicted_digit]

        label = f"Digit: {predicted_digit} ({max_conf*100:.1f}%)"

  

        # Bounding box & label

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)

        # Label Background

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(display_frame, (x1, max(y1 - 25, 0)), (x1 + text_w, max(y1, 25)), box_color, -1)

        # Text Overlay

        cv2.putText(display_frame, label, (x1, max(y1 - 5, 20)),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

  

    cv2.imshow("Pure YOLO Live Digit Tracker (0-9)", display_frame)

  

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

  

cap.release()

cv2.destroyAllWindows()


```