
# Act IX — The First Breakthrough: The Detector Could Actually See

> _“It wasn't working everywhere—but for the first time, I saw evidence that the detector was alive.”_

While testing the model, something unexpected happened.

The bounding box began appearing.

The model could identify the digit.

But only under a particular spatial condition.

It worked in a specific region of the frame.

And that was a huge moment.

Because the failure had changed character.

Previously, the model simply failed to recognize the digit.

Now I could see:

**it knew something was there.**

But it was strongly biased toward a particular position.

I had discovered **position bias**.

The detector was not yet learning the general concept:

> “A digit can exist anywhere.”

It was learning something closer to:

> “A digit that looks like this, at approximately this position and scale, is the thing I should detect.”

That was my first real experience of spatial bias inside a neural network.

And I immediately started asking why.

---

# Act X — The 2 A.M. Investigation: The Image Was Lying to the Network

> _“The model wasn't necessarily learning the wrong thing. I had accidentally shown it the wrong world.”_

I inspected what was actually entering the neural network.

I used visualization to see the processed images rather than trusting the preprocessing code blindly.

That revealed a major problem.

My camera images did not all have the same aspect ratio.

Many were captured using a phone camera.

They were not naturally square.

Yet I was squeezing them into a fixed square input.

That meant the image geometry itself was being distorted.

The digit's appearance and position were changing as a consequence of the resizing operation.

I had collected the data in different shapes, then forced those shapes into a common square representation without preserving their geometry correctly.

Augmentation had not completely solved the problem.

The network was therefore seeing a dataset with an unintended spatial pattern.

The position bias suddenly made much more sense.

This became another major lesson:

**data geometry is part of the learning problem.**

The network does not know that I accidentally distorted an image.

It simply learns from whatever I give it.

---

# Act XI — Padding Instead of Crushing the Image

> _“If the image cannot fit the square, don't crush it. Preserve it.”_

The solution I explored was padding.

Instead of aggressively squeezing every image into a square, I wanted to preserve the original aspect ratio and extend the shorter dimension.

The idea was:

**resize while preserving geometry → pad the remaining space → produce the final square input.**

This allowed images with different original shapes to reach the network in a more consistent spatial representation.

It was another important shift in my thinking.

I was no longer treating preprocessing as a cosmetic step.

I was beginning to understand that preprocessing defines the world the neural network sees.

I regenerated the pipeline.

I trained again.

And once again:

**failure.**

The problem was deeper than the image shape alone.
