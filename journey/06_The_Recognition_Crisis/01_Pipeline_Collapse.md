# Act I — The Pipeline Collapse

> _"Version 4 worked beautifully—but only inside its own artificial world."_

The OpenCV pipeline had finally become reliable under controlled conditions. Grayscale, Gaussian Blur, Otsu Thresholding, Contour Extraction, Morphological Cleanup, ROI cropping, and Softmax together produced a clean 28×28 image that closely resembled the MNIST dataset. Inside this controlled environment, the CNN predicted handwritten digits consistently.

The illusion broke the moment reality entered the frame.

A finger holding the paper immediately confused the pipeline. Small background changes or slight paper movement caused the prediction to collapse. At this stage, I still lacked the deeper CNN intuition that I would later gain through YOLO. I could observe the failures, but I could not yet fully explain them from the perspective of feature learning.

By debugging every stage individually, I identified the two dominant bottlenecks:

- **Otsu Thresholding**, whose global threshold became unstable whenever the finger or lighting altered the histogram.
- **Contour Extraction**, which depended entirely on Otsu's output. Once thresholding failed, contour localization also failed.

Ironically, these two components were both the pipeline's greatest strength and its greatest weakness. They perfectly transformed controlled images into MNIST-like inputs, but they completely destroyed robustness in real-world conditions.

The problem was no longer the classifier.

The problem had become the preprocessing pipeline itself.
