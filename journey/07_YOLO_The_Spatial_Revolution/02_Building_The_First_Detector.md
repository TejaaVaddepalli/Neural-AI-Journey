# Act V — Understanding the Full Mathematical Flow

> _“The architecture gave me the vision. The losses gave me the purpose. Backpropagation showed me how the entire system could learn.”_

Once I understood the forward process and the detection losses, I moved into backpropagation.

The basic mechanism itself was familiar.

The losses are combined.

The combined error flows backward.

Gradients propagate through the prediction layers.

Then through the convolutional layers.

Then through the earlier feature extraction stages.

Finally, the kernel weights are updated.

The important difference was what those gradients represented.

They were no longer correcting only:

**“You classified this digit incorrectly.”**

They were carrying information about:

- the target's spatial position,
    
- whether an object existed,
    
- the dimensions of the predicted box,
    
- and the identity of the object.
    

The individual losses accumulated into a larger learning signal.

That signal travelled backward through the network.

And this made my earlier CNN knowledge suddenly much more useful.

The convolutional network was still learning through gradients.

But now the gradients were teaching it a spatial task.

This was one of the strongest connections between my CNN stage and my YOLO stage.

---

# Act VI — The Theory Had to Become Code

> _“Understanding YOLO mathematically was one thing. Making my own detector actually work was something else entirely.”_

Eventually, the theory had to become implementation.

I wrote the YOLO implementation with AI assistance, as I had done throughout much of my journey.

I still could not comfortably write every large block of PyTorch code completely from memory.

So I used AI to generate code and then studied the generated structure, shapes, operations, and mathematical flow.

But there was a problem.

I was exhausted.

I had already spent a huge amount of time fighting the previous recognition system.

I wanted to move faster.

Because of that, I understood the generated code, but I did not inspect every assumption as deeply as I should have.

That decision would become one of the most important lessons of the entire YOLO stage.

Because the first implementation did not simply fail.

It failed in ways I did not immediately understand.

---

# Act VII — The Dataset Became the Next Battlefield

> _“YOLO could learn where the digit was—but first I had to tell it where the digit actually was.”_

YOLO introduced something my previous classifier did not fundamentally require in the same way:

**spatial labels.**

The network needed to know:

**Where is the digit?**

So the images had to contain bounding-box information.

I had already collected my first dataset for the binary experiment:

- **100 images of 0**
    
- **100 images of 1**
    

That meant roughly **200 digit images** for the initial detector attempt.

But now every image had to be labelled.

I used **LabelImg** to manually draw the bounding box around each digit and generate the corresponding YOLO `.txt` label files.

This was not a small detail.

It became a major part of the implementation itself.

I had to make sure that:

- the image corresponded to the correct label file,
    
- the class index was correct,
    
- the bounding-box coordinates were correct,
    
- the YOLO label format was correct,
    
- the paths were connected correctly,
    
- and the training pipeline could actually read those labels.
    

I manually labelled the images.

One after another.

Box after box.

File after file.

It was slow.

But after everything I had already experienced, I did not want to trust an automatic shortcut blindly.

I wanted to know exactly what the network was being told.

---

# Act VIII — YOLO Version 1: The First Real Implementation

> _“I finally had the theory, the labels, the architecture, and the training code. Surely this time it would work.”_

I trained the first YOLO version using the initial 0/1 dataset.

I also applied preprocessing.

At this stage, I experimented with the image representation rather than immediately rebuilding the entire sophisticated OpenCV pipeline.

Grayscale conversion became part of the input process.

I also tested inverted and non-inverted representations.

The idea was simple:

**give the network a cleaner representation of the digit and let the detector learn from it.**

I trained the model.

And waited.

Then the result came.

**Failure.**

Again.

The model did not suddenly become perfect simply because I had moved from CNN recognition to YOLO detection.

The architecture was more appropriate for the spatial problem, but the implementation still contained problems.

At that time, I could not immediately identify exactly what was wrong.

That was the frustrating part.

I understood the mathematics.

I understood the general architecture.

I understood what the loss was supposed to accomplish.

But the actual machine was still not behaving.

