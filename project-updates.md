---
permalink: /project-updates/
---

# Project Update 2
Jenna Brandt and Erin Puckett

## What have you completed or tried to complete?

- We have completed preparing a smaller version of our final dataset. Thus, we have taken the necessary steps to create a dataset which we can use to train a preliminary version of our model. We will go through these same steps again on the full, much larger dataset once we have finalized the neural network and modeling approach. The steps to clean our smaller dataset involved the following:

  - Downloading the datasets from Kaggle
  - Loading the csv files into a Jupyter notebook
  - Eliminating exccess data/columns in the dataset that are not useful for our purposes
  - Removing non-ascii characters
  - Dividing the data into classes
  - Shuffling the data in those classes
  - Selecting the first 1000 elements from each of the four shuffled classes
  - Concatenating all the chosen elements into a small dataset of 4000 items
  - Writing the small dataset to a new csv file

- Additionally, we have determined the steps we will take to create a neural network as follows:

  - Using fastai to create a preliminary model using the text classifier libraries
  - Then creating a neural network model from scratch using Pytorch textsentiment libraries


## What issues have you encountered?
So far, the only issue we encountered was a decision on whether to start immediately with Pytorch or to create a first (and hopefully quicker and easier) version of the model using fastai. We met with Prof. Clark who recommended that we start with fastai to get something working and then move on to a more in-depth approach to creating a more customized model for our dataset. We are looking forward to doing this.



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
