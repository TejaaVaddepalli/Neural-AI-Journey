# Act V — The Sliding Window Experiment

> _"If the whole image is too difficult, what if I search for the digit piece by piece?"_

The next idea came from the limitations I was observing.

When I held the digit close to the camera, the model could occasionally recognize it.

When I moved it farther away, the digit occupied a much smaller portion of the image and recognition became unreliable.

So I changed the problem.

Instead of sending the **entire camera frame** directly to the CNN, I created a **sliding window**.

The idea was simple:

**Take a small region → process it → send it to the CNN → move the region → repeat.**

The window travelled across the image from one position to another.

At each position, the selected region was converted into the format expected by the CNN and evaluated.

I experimented with the window size and the distance between consecutive positions.

A smaller stride gave the system more opportunities to encounter the digit.

A larger stride made the search faster, but increased the possibility of skipping important parts of the digit.

The experiment was an attempt to compensate for the model's inability to reliably recognize the digit when it occupied only a small portion of the complete frame.

---

## The New Failure

The sliding window improved the **search strategy**, but it exposed another problem.

The window did not always contain the complete digit.

Sometimes it captured only part of it.

Sometimes the digit was cut at the boundary of the window.

Sometimes the window contained too much background.

Sometimes the digit was present but represented at a scale or position that was still very different from the training examples.

The preprocessing pipeline could not reliably repair these cases.

The CNN was still fundamentally a recognition model expecting a suitable digit representation.

The sliding window was simply moving that recognition problem around the image.

I adjusted the window dimensions.

I changed the stride.

I increased and decreased the scanning density.

I experimented with different configurations to make sure the digit would be captured completely.

Still, the failures continued.

There were rare cases where the system produced a prediction in heavy background conditions, but it was not reliable enough to consider the problem solved.

The experiment had reached another boundary.

---

## The Recognition Boundary Becomes Clear

At this point, the entire evolution had exposed the same problem from different directions.

I had tried to make the **image** easier.

I had tried to make the **preprocessing** stronger.

I had tried to make the **dataset** larger.

I had tried to **fine-tune** the existing model.

I had trained the network entirely on my own data.

I had tried **augmentation**.

And finally, I had tried to **search the image using a sliding window**.

Yet the model still depended heavily on receiving a clean, appropriately positioned representation of the digit.

The sliding window could search.

The CNN could recognize.

But the complete system still could not robustly determine the digit's presence and position under arbitrary conditions.

That was the point where I reached the practical boundary of my recognition-based approach.

And a new question began forming:

> **Could the network itself learn the spatial location of the digit instead of forcing the entire problem onto preprocessing and window scanning?**

I did not begin this stage with a complete understanding of object detection.

I simply knew that the current architecture had reached its limit.

That question became the doorway to the next stage.

**YOLO.**