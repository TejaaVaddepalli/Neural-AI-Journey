import cv2

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import os

import os

  

if not os.path.exists('failures'):

    os.makedirs('failures')

    print("Created 'failures' folder for the mission!")

  

# 1. THE ARCHITECTURE

class NeuralAILabCNN(nn.Module):

    def __init__(self):

        super(NeuralAILabCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.dropout = nn.Dropout(0.25)

        self.fc = nn.Linear(32 * 7 * 7, 11)

        self.relu = nn.LeakyReLU(0.01)

  

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))

        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        return self.fc(x)

  

# 2. LOAD MODEL

model = NeuralAILabCNN()

checkpoint = torch.load('neural_empire_v4.pth')

model.load_state_dict(checkpoint['model_state'])

model.eval()

  

# 3. CAMERA

cap = cv2.VideoCapture(0)

x, y, w, h = 220, 140, 200, 200

  

stable_label = "Scanning..."

counter = 0

last_pred = None

prev_roi = None   # ✅ IMPORTANT

  

print("System Online. Press 'q' to quit.")

save_count = 0

  

# Initialize from existing files (only once)

existing = [name for name in os.listdir('failures') if 'fail' in name]

if existing:

    nums = [int(name.split('_')[1].split('.')[0]) for name in existing]

    save_count = max(nums) + 1

while True:

    ret, frame = cap.read()

    if not ret: break

  

    # =========================

    # STEP 1: ROI

    # =========================

    roi = frame[y:y+h, x:x+w]

  

    # =========================

    # STEP 2: CLEANING

    # =========================

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi_blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    roi_inverted = cv2.bitwise_not(roi_blurred)

  

    _, roi_thresh = cv2.threshold(

        roi_inverted, 0, 255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU

    )

  

    # =========================

    # STEP 3: CONTOUR FILTER

    # =========================

    contours, _ = cv2.findContours(

        roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE

    )

  

    roi_thresh1 = roi_thresh.copy()   # ✅ SAFE DEFAULT

  

    if contours:

        largest = max(contours, key=cv2.contourArea)

  

        mask = np.zeros_like(roi_thresh)

        cv2.drawContours(mask, [largest], -1, 255, -1)

  

        roi_thresh1 = cv2.bitwise_and(roi_thresh, mask)

  

        kernel = np.ones((3,3), np.uint8)

        roi_thresh1 = cv2.morphologyEx(roi_thresh1, cv2.MORPH_CLOSE, kernel)

        roi_thresh1 = cv2.dilate(roi_thresh1, kernel, iterations=1)

  

    # =========================

    # STEP 4: CENTERING → 28x28

    # =========================

    coords = cv2.findNonZero(roi_thresh1)

  

    if coords is not None:

        x_b, y_b, w_b, h_b = cv2.boundingRect(coords)

        digit = roi_thresh1[y_b:y_b+h_b, x_b:x_b+w_b]

  

        # MNIST STYLE (IMPORTANT)

        digit = cv2.resize(digit, (20, 20))

  

        canvas = np.zeros((28, 28), dtype=np.uint8)

        canvas[4:24, 4:24] = digit

  

        roi_resized = canvas

    else:

        roi_resized = np.zeros((28, 28), dtype=np.uint8)

  

    # =========================

    # STEP 5: SMOOTHING (FIXED)

    # =========================

    if prev_roi is None:

        prev_roi = roi_resized.copy()

  

    roi_resized = (0.7 * roi_resized + 0.3 * prev_roi).astype(np.uint8)

    prev_roi = roi_resized.copy()

  

    # =========================

    # STEP 6: BAD INPUT FILTER

    # =========================

    white_ratio = np.sum(roi_resized > 200) / (28*28)

    if white_ratio > 0.6:

        stable_label = "Invalid"

        cv2.imshow("D - Final 28x28", roi_resized)

        cv2.imshow("KING-Vision Live", frame)

        if key & 0xFF == ord('q'): break

        continue

  

    # =========================

    # STEP 7: NORMALIZE + PREDICT

    # =========================

    roi_final = roi_resized / 255.0

    roi_final = (roi_final - 0.1307) / 0.3081

  

    tensor_input = torch.FloatTensor(roi_final).view(1, 1, 28, 28)

  

    with torch.no_grad():

        output = model(tensor_input)

        probs = F.softmax(output, dim=1)

        conf, pred = torch.max(probs, 1)

  

    # =========================

    # STEP 8: STABILITY

    # =========================

    current_pred = pred.item()

  

    if current_pred == last_pred:

        counter += 1

    else:

        counter = 0

        last_pred = current_pred

  

    if conf.item() > 0.75 and counter > 5:

        stable_label = f"Digit: {current_pred} ({conf.item()*100:.1f}%)"

    elif conf.item() < 0.30:

        stable_label = "Scanning..."

    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'):

        cv2.imwrite(f"failures/fail_{save_count}.png", roi_resized)

        cv2.imwrite(f"failures/raw_{save_count}.png", roi)

        print(f"Captured Failure {save_count}: Pred was {current_pred} at {conf.item()*100:.1f}%")

        save_count += 1   # 🔥 IMPORTANT

  

    # =========================

    # STEP 9: DISPLAY

    # =========================

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, stable_label, (x, y-10),

                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('KING-Vision Live', frame)

    cv2.imshow("ROI", roi)

    cv2.imshow("Thresh", roi_thresh1)

    cv2.imshow("Final 28x28", roi_resized)

  

    if cv2.waitKey(1) & 0xFF == ord('q'): break

  

cap.release()

cv2.destroyAllWindows()
