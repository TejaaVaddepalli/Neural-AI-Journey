# Act II — Breaking the Pipeline

> _"If preprocessing is the bottleneck, remove preprocessing."_

My first instinct was not to retrain the network, but to simplify the vision pipeline.

I experimented with replacing Otsu using Fixed Thresholding and Adaptive Thresholding. Adaptive Thresholding introduced local artifacts and inconsistent digit thickness, while Fixed Thresholding failed under changing illumination. Neither produced images that resembled the MNIST distribution.

I then began removing individual preprocessing stages.

Contour Extraction was removed.

The ROI cropping logic was relaxed.

Eventually, I removed the ROI constraint entirely and allowed the model to observe much larger portions of the camera frame.

Every modification produced the same conclusion.

The classifier had never learned to understand real-world images.

It had only learned to recognize MNIST-style inputs.

Removing preprocessing did not improve robustness—it simply exposed how dependent the CNN was on the handcrafted pipeline.

This led to a much bigger realization.

The preprocessing pipeline was not the real bottleneck.

The training data was.