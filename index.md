# Tweet Classification and Comparison of Russian Propagandists and U.S. Politicians
Jenna Brandt and Erin Puckett

# PAPER DRAFT

## Introduction
During the last two presidential elections in the U.S., the impact of Russian disinformation on Twitter has created a need for effective solutions for finding and removing the trolls used to propagate disinformation. As such, we are investigating Twitter usage among Democratic and Republican politicians as well as tweets created by Russian trolls, as the trolls often try to imitate politicians’ rhetoric. We have created a deep learning model to effectively categorize the author of a tweet as either a Democrat, a Republican, a right-wing-imitating troll, or a left-wing-imitating troll. 

In addition to being extremely important, finding a solution to this issue is very difficult because there are many features of tweets that could be used to determine whether they were written by trolls or real people (message length, frequency, content, time tweeted, number of followers, retweets, etc.). Another approach could be to look at account-level details rather than tweet-level details. Furthermore, often those behind the trolls (especially those in cases like this of state-sponsored disinformation campaigns) closely study their targets in order to appear as similar as possible. There is an active intent to fool, and that makes tweets hard to tell apart, particularly as some U.S. politicians themselves have embraced disinformation. With all of that in mind, we decided to focus our algorithm on the content of the tweets themselves, rather than other features. 

We found that it is difficult to assess why our algorithm categorizes certain tweets as it does. The nature of the input with its varied structures might part of the reason - the algorithm might use the structure to categorize the tweet as politicians tend to have more structured tweets. That said, perhaps other solutions would better approach this issue. For example, troll tweets are shorter than politicians’ tweets, so perhaps instead of looking at the actual language used in the tweets, our algorithm might actually categorize tweets based on length, and then look at vocabularly to distinguish politicial affliation within the broader troll or politician categories. Regardless, our results using a content-based algorithm show significant success. 

Our algorithm correctly categorizes tweets as Democrat, Republican, LeftTroll, or RightTroll 84% of the time. Interestingly, the algorithm had more trouble distinguishing between Republican and Democratic tweets than between the politician and troll tweets, which points to the existence of some similarites between politicians of different parties and between trolls of different pretended affliations. This is surprising given the current state of political polarization. However, the significant level of success means that social media companies could try other approaches like ours to flag tweets that appear to come from sources who might intend to disrupt American democracy in some way, which would mean that the public (general users of social media) would not be subject to potentially very harmful messaging. 

When conducting this research, we kept in mind the ethical implications of categorizing tweets as “trolls” rather than real people. Freedom of speech is a common concern in today’s use of social media. Social media companies purport to be simply a platform for sharing one’s thoughts, and by classifying tweets as trolls, and thus potentially misclassifying real human tweets as trolls, we risk dismissing their thoughts as not real and part of a malicious campaign to harm American democracy. While the need to identify disinformation made this research worthwhile, we were aware of the potential negative implications of our classifications. 

## Related Works
There has been other work done by researchers involving machine learning and Twitter, specifically with a focus on politicians. We describe several closely related works as follows, and then we note how our research differs from others’ works. A University of Michigan and Georgia Tech [paper](https://arxiv.org/pdf/1901.11162.pdf) from 2018 focuses on classifying Russian trolls vs. “normal”/control accounts on Twitter. They pay particular attention to Russian attempts to interfere with the 2016 U.S. Election, and create a machine learning model to correctly predict Russian troll accounts from non-troll accounts with high accuracy. This paper, unlike ours, does not address American politicians specifically or make any distinctions between Republicans and Democrats.

Another [paper](https://arxiv.org/pdf/1802.04291.pdf), from USC researchers and presented at WWW in 2018, addresses misinformation. In this paper, the researchers used machine learning techniques to “investigate the role and effects of misinformation, using the content produced by Russian Trolls on Twitter as a proxy for misinformation.” They specifically looked at both liberal and conservative media outlets, and particularly focused on “users who re-shared the posts produced on Twitter by the Russian troll accounts publicly disclosed by U.S. Congress investigation” of 2016 election interference. This paper does not focus on actual politicians’ tweets but instead focuses on private citizen retweets; we will focus on the tweets of politicians.

An additional [paper](https://journals.sagepub.com/doi/pdf/10.1177/2158244019827715), from NYU and published in 2019, uses neural networks to classify tweets from Russian accounts as being pro-regime, anti-regime, or neutral. Specifically, the researchers used a “deep feedforward neural network (multilayer perceptron or MLP) that uses a wide range of textual features including words, word pairs, links, mentions, and hashtags to separate four contextually relevant types of trolls: pro-Kremlin, neutral/other, pro-opposition, and pro-Kiev.” The results were “high-confidence predictions for most observations”. This paper focuses on only Russian accounts and does not investigate American politicians, compared to our work which concentrates on Russian-produced tweets that relate to events in United States politics as well as American politicians.

Finally, a [paper](https://arxiv.org/pdf/1802.04289.pdf) from 2018 involving researchers from USC and the Indian Institute of Technology uses neural networks with a contextual long short-term memory (LSTM) architecture that looks at both content and metadata from Twitter accounts to identify trolls among real human users. The authors also used synthetic minority oversampling in order to create a large dataset for training. Our paper will address a similar issue involving classification of troll tweets versus real human user tweets, but with a focus on political speech involving Russian trolls and American Democrat and Republican politicians. Thus, there are a variety of papers that have investigated political tweets, troll tweets, Russian-produced tweets, and differences between troll and real human tweets, but none have specifically looked to distinguish Russian trolls, American Republican politicians, and American Democrat politicians. We hope this paper brings some insight into the area of political tweets and Russian interference in American political discourse on Twitter.

# Methods Outline
The software we used for this project involved Python libraries, namely fastai and Pytorch. We used fastai to create a preliminary model with a AWD LSTM architecture and a dropout magnitude of 0.5. Then focused on using Pytorch for the remainder of the project. Pytorch allowed us to more fully customize our model and get an in-depth understanding of how to create a neural network for a natural language processing application involving political tweets. The Pytorch models involved a “bag of words” approach as well as word embeddings to approximate the mathematical meaning of each written tweet.

First, we accessed two datasets from Kaggle, the “Russian Troll Tweets” dataset from FiveThirtyEight and “Democrat Vs. Republican Tweets'' from Kyle Pastor. After loading the data into a Jupyter notebook, we eliminated excess columns and irrelevant row (i.e. Russian Troll tweets not classified as RightTroll or Left Troll. Once that was completed, the data was cleaned to remove non-ascii characters. We then divided into classes, we shuffled it and selected the first 1000 items in each class. Once concatenated together, we had our preliminary small dataset with which we started building our models. Following the use of the small dataset to build our models, we went back using the same data cleaning methods, but this time used the entirety of the cleaned dataset, resulting in a final dataset of size X, with X RightTroll entries, X LeftTroll entries, X Republican entries, and X Democrat entries.  

The tools we used for data analysis were matplotlib, seaborn, and other data visualization libraries. This allowed us to take our various versions of the Pytorch neural network classification model and visualize the results to compare between the models. In creating the models themselves, we implemented fastai’s text classifier model as well as Pytorch’s text sentiment library.

# Discussion Outline
Both our fastai model and our Pytorch model yielded significant results. We will present the accuracies and validation losses of those models as well as the changes we made to try to increase the accuracies and decrease those losses. We will also discuss what our results may mean in a broader geopolitical context.

We will interpret and evaluate our data by comparing the validation accuracies of each of the four classes from both models as well as by using a confusion matrix described as follows. The confusion matrices for both models show that the models have the greatest difficulty distinguishing between Democrat and Republican tweets and between LeftTroll and RightTroll tweets. This was surprising and heartening to us, as it shows that the Russian propagandists were mostly unsuccessful in emulating U.S. politicians, and that there is perhaps a smaller divide between the two parties, at least in terms of how they structure or word their tweets. 

Compared to others, our work investigates a specific inquiry as to how well a NLP model can distinguish between actual politician tweets and Russian troll tweets, divided by Democrat/Republican and RightTroll/LeftTroll. Other researchers have looked at Russian troll tweets but not in the context of comparing them to American politicians. Because we found that our NLP model can quite accurately determine American politician tweets from Russian troll tweets, our work demonstrates that social media companies may be able to use a model such as ours to flag and/or delete tweets that are likely to come from troll accounts to limit the influence of foreign governments on American politics.

Our claim is that deep learning can be used to predict American politician tweets versus Russian troll tweets, and we demonstrate that our claim has merit by presenting the results of our model which, with high accuracy, distinguish between different classes of American politicians and trolls. The magnitudes of the accuracies of the models as well as the trends in the confusion matrices show that deep learning can be successful in identifying trolls when compared to politicians on Twitter.

Further steps we could take would be to look at the troll tweets and compare them to political tweets not written by politicians. As this has somewhat been done before, it would be interesting to see our model, trained on politician tweets as the ‘non-troll’ tweets, would fare compared to other models trained on a broader set of political troll and human tweets.

## Bibliography

A. Badawy, E. Ferrara and K. Lerman, "Analyzing the Digital Traces of Political Manipulation: The 2016 Russian Interference Twitter Campaign," 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), Barcelona, Spain, 2018, pp. 258-265, [https://doi.org/10.1109/ASONAM.2018.8508646](https://doi.org/10.1109/ASONAM.2018.8508646)

Im, J., Chandrasekharan, E., Sargent, J., Lighthammer, P., Denby, T., Bhargava, A., Hemphill, L., Jurgens, D., & Gilbert, E. (2020). Still out there: Modeling and Identifying Russian Troll Accounts on Twitter. 12th ACM Conference on Web Science. [https://doi.org/10.1145/3394231.3397889](https://doi.org/10.1145/3394231.3397889)

Kudungunta, S. & Ferrara, E. (2018). Deep Neural Networks for Bot Detection. Cornell University. [https://doi.org/10.1016/j.ins.2018.08.019](https://doi.org/10.1016/j.ins.2018.08.019)

Stukal, D., Sanovich, S., Tucker, J. A., & Bonneau, R. (2019). For Whom the Bot Tolls: A Neural Networks Approach to Measuring Political Orientation of Twitter Bots in Russia. SAGE Open. [https://doi.org/10.1177/2158244019827715](https://doi.org/10.1177/2158244019827715)



# PROJECT UPDATES

# Project Update 2 (for 3/24/21)

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


# Project Update 1 (for 3/3/21)

## Software we will use:
We will use PyTorch, linked [here](https://pytorch.org/), specifically the TextSentiment model.

## Dataset we will use:
We will use both a Russian trolls tweets [dataset](https://www.kaggle.com/fivethirtyeight/russian-troll-tweets) as well as a [dataset](https://www.kaggle.com/kapastor/democratvsrepublicantweets) that contains tweets of Democratic and Republican politicians.

## Overview of neural network specifications:
- _Type of neural network:_ We will be using the PyTorch TextSentiment model, which is a recurrent neural network.
- _Shape and type of inputs:_ The input dataset will contain tweets with labels corresponding to their classes (Russian, Democrat, or Republican). The shape of the input will vectors corresponding to words which converted via embedding. The size of the word embedding vector can be determined; we will start with a embedding dimension of 10. We will also use the size of the total number of words in the dataset in creating the embedding table, which will have size V x D, where V is the vocabulary size and D is the embedding dimension.
- _Shape and type of outputs:_ The outputs will be the predicted classes, Russian, Democrat, or Republican, for each tweet. We are doing classification, so the shape of the output for a single input image will be a 3 x 1 vector corresponding to the likelihood of each predicted class.
