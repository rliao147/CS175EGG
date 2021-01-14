---
layout: default
title: Proposal
---
## Summary of the Project

Given halves of images, our AI will complete the image by creating a full pixel art of the image in Minecraft. Our AI will specialize in creating images of a specific subject (such as ice cream, subject to change). It will train on a database of images of this subject, learning how each complete image looks like. It will then be able to use this knowledge to finish constructing new halves of images that it receives.

For example, if we were to give our AI half of an ice cream sandwich, it will determine that the type of ice cream in the image is “ice cream sandwich”, then finish the image based on its previous understanding of how ice cream sandwiches look.

Image completion/prediction is useful in augmented reality applications. For example, when placing a realistic augmented reflective item into the world, the item must reflect parts of the world it can not see. Image prediction can be used to estimate that part. Image prediction can also recreate faces of people when we only see a portion of their faces, this can be helpful for law enforcement. Image prediction can also help us recreate damaged photos with historical or sentimental importance.

## AI / ML Algorithms

We anticipate that our image completion project will require the use of: supervised learning / deep learning with images (separating the database images into groups and learning about them), decision trees or something similar (to classify the new half of the image into one of the groups), and a neural network, potentially a DCGAN or GAN (predict the other half of the given image).

## Evaluation Plan

Our team will evaluate the success of our project based on how accurate our agent predicts the other half of the given image. Our baseline evaluation is to consider whether or not the generated photo accurately reflects what the object was supposed to be. For example, if we specified to our agent that the given half was a photo of ice cream, we would expect the output of the other half of the photo to complete an image of ice cream. We’ll evaluate our data based on how accurate our generated photo matches the given photo (a percentage accuracy evaluation of the pixel colors for both images, with some room for error to account for Minecraft’s limited block color selection). 

Our original sanity cases would be a perfectly symmetrical image. If the agent is able to recognize that the half of a photo is supposed to complete a symmetrical image, that is a concrete milestone towards our moonshot scenario. Considering that there are many types of ice cream, it would also be a milestone for our agent to recognize which category of ice cream that it assumes the photo to be (soft serve, popsicles, scoops of ice cream). This classification will be logged or printed in the commands. Our moonshot case would be to match the given photo to near perfection (without seeing the other half!), along with perfectly identifying the type of ice cream given the half of a photo.

## Appointment with Instructor

Time and date of meeting time with instructor: 2:15 - 2:30PM, Thursday, January 21, 2021.
