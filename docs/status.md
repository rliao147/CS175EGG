---
layout: default
title:  Status
---

# {{ page.title }}


## Project Summary:

Given parts of images, our AI will predict and complete the image. Our AI will specialize in finishing images of minecraft biomes (such as landscapes of different biomes and buildings in those biomes). It will train on a database of images of this subject, learning how each complete image looks like. It will then be able to use this knowledge to finish constructing the remaining parts of images that it receives. 

For example, if it receives the left half of an image, it will recreate the right half, and vice versa. Moreover, if it receives an image with the center missing, then it will fill in the center. The images will be filled in based on our AI’s understanding of how Minecraft biome landscapes generally look like (based on it’s training dataset).  

Image completion/prediction is useful in augmented reality applications. For example, when placing a realistic augmented reflective item into the world, the item must reflect parts of the world it can not see. Image prediction can be used to estimate that part. Image prediction can also recreate faces of people when we only see a portion of their faces, this can be helpful for law enforcement. Image prediction can also help us recreate damaged photos with historical or sentimental importance.


## Approach:

## Evalution:

## Reamining Goals and Challenges:

Our goal for the next four weeks is to have a finalized image completion software that is able to complete masked images using various masks upon different biomes. Currently, we are facing hardware constraints, as the training of images larger than 512x512 pixels is very inefficient. If we can figure out a solution to speed up our training data and require less hardware usage, it will be a lot easier to generate higher quality images. We have an efficient method of gathering training data, and training different biomes should be more or less the same. The next step is to move on to phase two of training our GAN, which is to recognize masked images (instead of generating images from scratch). This would require our generator to generate realistic images to fool our discriminator and have the unmasked pixels of the input closely represent those pixels in the output.

## Resources Used:

https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/

https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/

http://keras.io/api/preprocessing/image/

https://towardsdatascience.com/writing-your-first-generative-adversarial-network-with-keras-2d16fd8d4889

https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
