# Physics-Informed Super-Resolution for Turbulent Flows

## Model

The upsampling model is based on a GAN with a similar architecture as models used for super-resolution of images.

The generator is a convolutional neural network with redidual blocks and upsampling layers based on pixel shuffling. The discriminator has a convolutional architecture, follows by fully connected layers for binary classification.
