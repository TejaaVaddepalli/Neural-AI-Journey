# Act III — Teaching the Network a New World

> _"If reality doesn't resemble MNIST, perhaps MNIST should no longer define reality."_

Instead of forcing every camera image to imitate MNIST, I decided to change the training data itself.

I began building my own handwritten dataset.

The first version consisted of **matrix sheets**, where a single page contained dozens of handwritten versions of the same digit. Every digit was photographed individually using my own camera and passed through multiple preprocessing pipelines before training.

Each preprocessing method was evaluated independently.

- Otsu Thresholding frequently destroyed thin strokes.
- Adaptive Thresholding generated noisy local patterns.
- Fixed Thresholding lacked robustness under varying illumination.
- Eventually I began experimenting with minimal preprocessing instead of aggressive thresholding.

I attempted transfer learning through fine-tuning.

Initially only the final layers were updated while earlier layers remained frozen.

When that failed, progressively more layers were unfrozen.

Finally, the entire network was trained.

None of these approaches produced meaningful improvement.

The network continued struggling because the newly collected dataset remained extremely small compared to the diversity required for generalization.

The bottleneck had shifted once again.

The limitation was no longer preprocessing.

It was dataset diversity.