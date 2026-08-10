# Act IV — Breaking the Recognition Boundary

> _"I had changed the data. I had changed the preprocessing. Now I had to find out whether the network itself could adapt."_

The first attempt was **fine-tuning**.

The idea was straightforward: the CNN already understood handwritten digits through MNIST, so perhaps I only needed to adapt its learned representations to my camera data.

I began with the later layers.

The earlier layers were frozen while the final layers were allowed to train on my newly collected handwritten samples.

It failed.

The training did not produce the improvement I expected. The loss did not decrease meaningfully; in some experiments it became worse instead of better. The network was not gradually adapting to the new distribution in a useful way.

I changed the amount of trainable layers.

I unfroze additional layers.

I tried training deeper portions of the network.

Eventually, I removed the freezing completely and allowed the entire CNN to train.

**Still nothing.**

The loss behavior remained poor, and the model did not develop reliable recognition of the new camera images.

Fine-tuning had reached its limit.

---

## Increasing the Dataset

I then questioned the dataset itself.

The initial matrix-sheet approach gave me many handwritten examples, but the actual diversity of the training set was still limited.

So I started collecting more images for every digit.

I moved progressively from the tiny initial dataset toward larger collections—around **30, then 50 or more samples per digit**—while introducing controlled variations in:

- handwriting style
- stroke thickness
- lighting
- background conditions
- image positioning
- small visual disturbances

I also experimented with augmentation to create additional variations from the collected samples.

But increasing the number of images alone was not enough.

The representation entering the CNN still mattered.

---

## The Preprocessing Experiments

I generated different versions of the collected data and trained them separately rather than assuming one preprocessing method would solve everything.

I revisited:

**Gaussian Blur → Otsu Thresholding → Fixed Thresholding → Adaptive Thresholding → Morphological processing → minimal preprocessing**

Otsu frequently produced broken or unusable digit representations.

Fixed thresholding could not adapt reliably to illumination.

Adaptive thresholding behaved differently at the small spatial scale of the extracted images, producing local artifacts and changing the appearance of the handwritten strokes.

I began inspecting the resulting **28×28 representations directly**.

If an image was broken, distorted, excessively noisy, or no longer visually representative of the digit, I removed it from the training set.

This gave me much more control over what the network was actually seeing.

But the fundamental problem remained.

---

## Abandoning MNIST

At this point, I made a much larger change.

Instead of continuously trying to make my camera images resemble MNIST, I removed the original MNIST training distribution from the experiment.

I trained the network on **my own collected handwritten dataset**.

This was no longer fine-tuning.

It was a complete change in the training distribution.

The model now had to learn directly from the visual characteristics of my own camera data.

There was some improvement.

When the digit was presented close to the camera and under favorable conditions, the model could occasionally recognize it.

That small success was important.

It proved that the network was not completely incapable of learning the new representation.

But the improvement was extremely limited.

Move the digit farther away.

Change its position.

Introduce more background.

Change the appearance of the paper.

Alter the way the digit was presented.

The recognition quickly deteriorated.

The model could recognize **some examples**.

It could not reliably generalize to **the environment**.

And this distinction became the central problem.

I had spent the previous experiments removing preprocessing bottlenecks and rebuilding the dataset.

Now I had another boundary:

> **The CNN could recognize the digit when the input was favorable, but it had no reliable mechanism for finding and recognizing that digit across the entire camera frame.**

I needed another experiment.