# Class-level image errors and seed variation

This is a complete local image workflow using 32 authored, non-personal 12x12 RGB PNG patterns in four classes: cloudy, rainy, sunny and sunrise. They resemble simple weather symbols and are not photographs. The small renderer can create shortcuts; results demonstrate procedure and do not establish real weather recognition.

## Supplied data and split

metadata.csv names every image, class, source group, partition and exact PNG SHA256. Four source families per class each contain two related variants. Two whole families train, one validates and one tests: 16/8/8 images, with every class in every set. Related variants stay together. All families still share the same four drawing rules: separating named originals does not create a renderer-independent or real-photo test. Each role contains all 32 original PNGs and four text files (36 files total), fitting the site's existing project limit.

At runtime the program verifies checksums and materializes separate temporary training/validation/test class directories from this manifest before decoding. Those directories are reproducibly regenerated from the source originals. The strict standard-library decoder supports only these generated, non-interlaced RGB8/filter-zero PNGs, caps the decompressed byte count, and verifies CRCs. It is not a general image loader for arbitrary photos.

## Implement the starter

Complete image_metrics in model.py. Compute the actual-row/predicted-column 4x4 confusion matrix,accuracy,mean class F1,per-class recall and class support. Zero denominators contribute zero. Compare every class and both fixed seeds;inspect an error from each class when present and record when no error is observed.

The CNN performs a real learned 3x3 valid RGB convolution with two shared filters, tanh activations, 2x2 average pooling, flattening to 50 values, and a four-class softmax. It has 260 trainable parameters. Its explicit minibatch gradients train convolution, output weights and biases. A separate flattened-pixel softmax has 1,732 parameters and provides a dense comparison. Neither model is a pretrained recognizer.

Establish the training-majority and mean-color prototype baselines. Fit color prototypes on training images only. Train the two neural architectures with fixed seeds 7 and 19; apply flip/brightness augmentation to training batches only. Preserve validation and test bytes. Validation cross-entropy retains each model's earliest best checkpoint, with patience two. Candidate choice uses mean validation macro F1 and the declared simpler preference within 0.02. A neural seed is selected by validation loss. Freeze every choice before opening held-out labels.

## Bound training and evaluate

Defaults are six epochs, batch four and a shared eight-second training CPU budget. CLI limits are at most twelve epochs, batch eight and twelve training CPU seconds. Check the clock before every bounded minibatch; final validation/test evaluation is additional bounded work. Every model reports actual updates, completed epochs, checkpoint and stop reason. An exhausted budget can legitimately return the initial checkpoint and must not be described as completed training.

Report held-out accuracy, macro F1, four-class recall/support and confusion matrices, both seed results, architecture and resource use. Inspect the error filenames and generated contact sheet. If a class has no observed errors, state that fact rather than inventing an example. Compare against the simpler baselines; the CNN need not win.

Hand check: four equally likely class probabilities give loss log(4). For the true class, the logit gradient is -0.75 and each other class is +0.25. A pooled value is the average of four convolution activations, so each receives one fourth of the upstream gradient. The repository's independent finite-difference checks verify the actual shared-kernel and output derivatives.

Use Python 3.10+ with its standard library. Run python3 main.py from any directory; --epochs, --batch-size and --seconds stay within the caps above. Optional --output report.json writes full metrics, prediction CSV, self-contained image/contact SVG and model card. Restart from the source to reproduce the splits, weights/checkpoints and predictions when the same budget completes. CPU timing itself varies.

Historical Colab/Mendeley weather examples are a separate source audit. This authored implementation preserves the four-class neural-image task without external downloads, TensorFlow, learner photos or personal data.

CNN reference: https://www.tensorflow.org/tutorials/images/cnn

Role: starter.
