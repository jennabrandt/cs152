---
permalink: /project-update-1/
---

# Project Update 1
Jenna Brandt and Erin Puckett

## Software we will use:
We will use PyTorch, linked [here](https://pytorch.org/), specifically the TextSentiment model.

## Dataset we will use:
We will use both a Russian trolls tweets [dataset](https://www.kaggle.com/fivethirtyeight/russian-troll-tweets) as well as a [dataset](https://www.kaggle.com/kapastor/democratvsrepublicantweets) that contains tweets of Democratic and Republican politicians.

## Overview of neural network specifications:
- _Type of neural network:_ We will be using the PyTorch TextSentiment model, which is a recurrent neural network.
- _Shape and type of inputs:_ The input dataset will contain tweets with labels corresponding to their classes (Russian, Democrat, or Republican). The shape of the input will vectors corresponding to words which converted via embedding. The size of the word embedding vector can be determined; we will start with a embedding dimension of 10. We will also use the size of the total number of words in the dataset in creating the embedding table, which will have size V x D, where V is the vocabulary size and D is the embedding dimension.
- _Shape and type of outputs:_ The outputs will be the predicted classes, Russian, Democrat, or Republican, for each tweet. We are doing classification, so the shape of the output for a single input image will be a 3 x 1 vector corresponding to the likelihood of each predicted class.
