
> _“I had one final attempt left. I collected more data, rebuilt the training set, checked the implementation again, and this time the detector finally worked.”_

By the time I reached July 9, I was exhausted.

I had already spent a long time trying to make the detector work.

The problem was no longer that I did not understand what YOLO was supposed to do. I understood the architecture, the spatial grid, the target representation, the losses, the convolutional flow, and the purpose of learned localization.

The problem was making **my implementation** actually behave according to that mathematics.

And I had made one important mistake in how I approached the previous implementation.

I had used AI assistance to generate parts of the code. Because I was exhausted, I did not investigate the generated implementation with the depth I normally would. I glanced through it, understood the general structure, and moved forward.

That did **not** mean I was treating the model as a black box.

I knew what the network was doing.

I knew the mathematical concepts.

I knew the architecture.

The mistake was much simpler and much more dangerous:

**I trusted generated implementation details without verifying them deeply enough.**

That became the real engineering lesson.

AI-generated code can look completely reasonable and still contain an implementation error.

So I stopped treating generated code as something that could simply be accepted.

I had to verify it.

---

## The Final Attempt

I decided to make one final serious attempt.

Instead of continuing with the smaller dataset, I collected my own additional images.

The original binary dataset had been too limited for the variation I wanted the detector to learn.

So I increased the dataset.

I collected approximately:

- **200 images of 0**
    
- **200 images of 1**
    

along with background images for the detector to learn when no digit was present.

I then manually prepared and labelled the new images.

I used **LabelImg** to draw the bounding boxes and generate the corresponding YOLO label files.

I did not simply accept automatically generated boxes.

I located the digit myself.

I drew the bounding box myself.

I checked the corresponding labels.

The dataset itself became part of the debugging process.

And I did all of this while I was also dealing with a fever.

At this point, I was genuinely running on the feeling of:

> **One last attempt.**

If this failed again, I had to decide whether continuing to fight the binary detector was still worth the time.

---

## The Implementation Had to Be Questioned

This time, I went back into the implementation rather than assuming that the generated code was correct.

I checked the actual data flow.

I checked the image paths.

I checked the label files.

I checked the target matrices.

I checked the tensor dimensions.

I checked the diagnostic visualizations.

I checked the prediction grid.

And the failures finally became concrete.

### The Ghost Target

The dataset parser was not correctly connecting the image files with their corresponding target information.

The result was that the network could receive an image while the target representation did not correctly describe the object in that image.

The learning system therefore did not always have the target it was supposed to learn from.

The image-label connection had to be explicitly verified.

Once corrected, the target matrix actually contained the object information that the loss function needed.

This became the **Ghost Target** problem.

---

### The Diagnostic Indexing Error

The debugging process exposed another problem.

My diagnostic visualization itself had an indexing mistake.

I intended to inspect the objectness channel across the entire 7 × 7 grid.

But the tensor was being sliced incorrectly.

Instead of extracting the channel dimension, the diagnostic code selected a spatial dimension.

That made the visualization suggest that the target information was missing even though the target itself was present.

The visualization was wrong because **my indexing was wrong**.

The correction was to select the objectness channel correctly across the spatial grid.

That gave me a much more reliable view of what the detector was actually receiving.

The lesson was immediate:

> **A diagnostic is only as trustworthy as the tensor indexing behind it.**

---

### The Shape Misalignment

Then came the deepest implementation problem.

The convolutional output naturally followed the PyTorch layout:
$$
[Batch, Channels, Height, Width]$$

My manually constructed target followed:

$$[Batch, Height, Width, Channels]$$

The dangerous part was that both contained the same numerical dimensions:

$$7 × 7 × 7$$

So the shapes looked correct.

But their meanings were completely different.

The loss was therefore comparing information according to the wrong dimensions.

The model's channels and the target's spatial dimensions were being interpreted against each other.

The numbers existed.

The dimensions existed.

But the **meaning of the dimensions did not agree**.

That was the real structural failure.

---

## The `.permute()` Correction

The solution was to explicitly realign the model output:

**[B, C, H, W]**

into:

**[B, H, W, C]**

using:

```python
permute(0, 2, 3, 1)
```

The values themselves were not being changed.

Their arrangement was being reorganized so that the prediction tensor and target tensor spoke the same dimensional language.

Now the loss could correctly interpret:

- channel 0 → objectness
    
- channels 1–4 → bounding-box information
    
- remaining channels → class information
    

This was the point where the implementation finally became mathematically aligned.

The engineering retro records this tensor-layout correction as one of the decisive fixes in the final detector implementation.

---

## Adaptive Thresholding Became Part of the Final Pipeline

I also reconsidered the image representation.

For the final binary detector, I returned to **adaptive thresholding**.

The goal was not simply to make the image look cleaner.

The goal was to give the network a consistent visual representation in which the digit remained distinguishable from its surroundings even when illumination changed.

The same preprocessing had to be used during training and inference.

The training images and the live camera frames needed to belong to the same visual distribution.

The final pipeline therefore became:

**camera frame → square padding → adaptive threshold → 448×448 input → Pure YOLO → 7×7 confidence grid → highest-confidence cell → bounding-box reconstruction → digit classification**

The implementation itself confirms the synchronized square-padding and adaptive-threshold preprocessing used before the network.

---

## July 9 — It Finally Worked

After correcting the dataset-target connection, the diagnostic indexing, the tensor-layout mismatch, and the preprocessing pipeline, I trained the detector again.

This time, the behavior changed.

The confidence map began developing localized responses.

The prediction grid began corresponding to the target grid.

The detector began identifying where the digit actually existed.

And finally:

# **The 0/1 detector worked.**

That happened on **July 9, 2026**.

This was not simply a moment where the model produced a good prediction.

It was the moment when the entire pipeline finally became coherent:

**data → target → tensor layout → loss → gradients → spatial prediction → bounding box**

The detector was no longer dependent on a manually selected region of interest.

The network itself was learning the spatial problem.

That was the reason I had entered YOLO in the first place.

---

## What July 9 Actually Taught Me

The most important lesson was not:

> “AI gave me wrong code.”

It was:

> **AI-generated code must never be trusted blindly.**

AI assistance was useful.

It helped me generate implementation quickly.

But generation was not verification.

I had to become responsible for checking:

- the data,
    
- the labels,
    
- the tensor shapes,
    
- the indexing,
    
- the target representation,
    
- the loss,
    
- and the actual behavior of the network.
    

That did not turn my model into a black box.

It did the opposite.

It forced me to inspect the machine more deeply.

By July 9, I had learned something that would remain important far beyond YOLO:

**Knowing the mathematics is not enough if the implementation does not preserve that mathematics.**

The detector working was the visible result.

The ability to diagnose why it had failed was the deeper achievement.

---

## The Boundary

The 0/1 detector was not the final version of my YOLO journey.

But it was the boundary I needed.

Before this point, I had been fighting to make a system recognize the right image.

Now I had crossed into something different:

**the network could learn where the information was.**

That was the real July 9 breakthrough.

**I had moved from recognition toward learned spatial understanding.**