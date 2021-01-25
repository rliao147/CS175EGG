---
layout: default
title: Proposal
---
## Summary of the Project

Given halves of images, our AI will predict and complete the image. Our AI will specialize in finishing images of minecraft biomes (such as landscapes of different biomes and buildings in those biomes). It will train on a database of images of this subject, learning how each complete image looks like. It will then be able to use this knowledge to finish constructing new halves of images that it receives.

For example, if we were to give our AI half of a desert biome, it would then finish the image based on its previous understanding of how images of desert biomes look.

Image completion/prediction is useful in augmented reality applications. For example, when placing a realistic augmented reflective item into the world, the item must reflect parts of the world it can not see. Image prediction can be used to estimate that part. Image prediction can also recreate faces of people when we only see a portion of their faces, this can be helpful for law enforcement. Image prediction can also help us recreate damaged photos with historical or sentimental importance.

## AI / ML Algorithms			
We anticipate that our image completion project will require the use of: supervised learning / deep learning with images (separating the database images into groups and learning about them), decision trees or something similar (to classify the new half of the image into one of the groups), and a neural network/pixel to pixel network, potentially a DCGAN or GAN (predict the other half of the given image).
## Evaluation Plan

Our team will evaluate the success of our project based on how accurate our agent predicts the other half of the given image. Our baseline evaluation is to consider whether or not the generated photo accurately reflects what the object was supposed to be. For example, if we specified to our agent that the given half was a photo of a desert dunes in minecraft, we would expect the output of the other half of the photo to complete an image of the desert dunes. We’ll evaluate our data based on how accurate our generated photo matches the given photo (a percentage accuracy evaluation of the pixel colors for both images, with some room for error to account for how Minecraft’s landscape can differ greatly). 

Our original sanity cases would be a perfectly symmetrical image of a simple biome, such as the ocean biome. If the agent is able to recognize that the half of a photo is supposed to be an ocean and sky, that is a concrete milestone towards our moonshot scenario. Considering that there are many types of Minecraft biomes, it would also be a milestone for our agent to recognize which biome that it assumes the photo to be (beach, desert, jungle, taiga). This classification will be logged or printed in the commands. Our moonshot case would be to recreate the given photo to near perfection (without seeing the other half!), as it would need to perfectly identify the type of biome given the half of a photo.

## Sprint Schedule (3 weeks)
Goal: Finish training agent for the beach biome.
ToDo: 
Create an XML file that sets a certain world specific to one biome.
Look into how worlds are generated based on XML files
Create a script that has an agent walking through the biome and takes screenshots in order to generate training data.
Create training module for algorithm.
Follow general image completion tutorial and adapt to our use case
Gan, pixel to pixel network 
Status report

## Team Meeting Schedule
Monday, Thursday 3:00pm

## Appointment with the Instructor
Thursday 01/21 2:15pm 
