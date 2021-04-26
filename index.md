# Tweet Classification and Comparison of Russian Propagandists and U.S. Politicians
Jenna Brandt and Erin Puckett

# PAPER DRAFT

## Introduction
During the last two presidential elections in the U.S., Russian disinformation has spread on Twitter. This has created a need for an effective way to find and remove the trolls used to propagate that disinformation. As such, we are investigating Twitter usage among Democratic and Republican politicians as well as tweets created by Russian trolls, as the trolls often try to imitate politicians’ rhetoric. We have created a deep learning model to effectively categorize the author of a tweet as either a Democrat, a Republican, a right-wing-imitating Russian troll, or a left-wing-imitating Russian troll.  

In addition to being extremely important, finding an effective solution to this issue is difficult because there are many features of tweets that could be used to determine whether they were written by trolls or real people, where both groups are divided by political affliation, including message length, frequency, content, time tweeted, number of followers, and retweets. Another approach could be to look at account-level details rather than tweet-level details. Often, those behind the trolls (especially those in cases like this of state-sponsored disinformation campaigns) closely study their targets in order to appear as similar as possible. There is an active intent to fool, and that makes tweets hard to tell apart, particularly as some U.S. politicians themselves have embraced disinformation. With all of that in mind, we decided to focus our algorithm on the content of the tweets themselves, rather than other features. 

We found that it was difficult to assess why our model categorizes certain tweets as it does. The nature of the input (the content of the tweet) with its varied length, structure, and inclusion or lack thereof of hashtags offers a variety of features for the model to rely on to distinguish between the classes. That said, perhaps focusing specifically on one or two feature is how the model works. For example, troll tweets seem to be shorter and less structured than politicians’ tweets, just from a glance through the dataset. Thus, perhaps instead of looking at the actual language used in the tweets, our model might actually categorize tweets based on length, and then look at vocabulary to distinguish politicial affliation within the broader troll or politician categories. Regardless of how the model is working, our results show significant overall success. Thus, we argue that deep learning can be effectively used to root out trolls on social media, partcularly Twitter. 

Our model correctly categorizes tweets as Democrat, Republican, LeftTroll, or RightTroll 81.5% of the time. Interestingly, the model had more trouble distinguishing between Republican and Democratic tweets than between the politician and troll tweets, which points to the existence of some similarites between the tweets of politicians of different parties and between the tweets of trolls of different pretended affliations. This is surprising given the current state of political polarization. However, the significant level of success means that social media companies could try other approaches like ours to flag tweets that appear to come from sources who might intend to disrupt American democracy in some way, which would mean that the public (general users of social media) would not be subject to the potentially very harmful messaging of Russian trolls, or trolls more generally. 

When conducting this research, we kept in mind the ethical implications of categorizing tweets as “trolls” rather than real people. Freedom of speech is a common concern in today’s use of social media. Social media companies purport to be simply a platform for sharing one’s thoughts, and by classifying tweets as trolls, and thus potentially misclassifying real human tweets as trolls, we risk dismissing their thoughts as not real and part of a malicious campaign to harm American democracy. While the need to identify disinformation made this research worthwhile, we were aware of the potential negative implications of our classifications. 

## Related Works
There has been other work done by researchers involving machine learning and Twitter, specifically with a focus on politicians. We describe several closely related works as follows, and then we note how our research differs from others’ works. A University of Michigan and Georgia Tech [paper](https://arxiv.org/pdf/1901.11162.pdf) from 2018 focuses on classifying Russian trolls vs. “normal”/control accounts on Twitter. They pay particular attention to Russian attempts to interfere with the 2016 U.S. Election, and create a machine learning model to correctly predict Russian troll accounts from non-troll accounts with high accuracy. This paper, unlike ours, does not address American politicians specifically or make any distinctions between Republicans and Democrats.

Another [paper](https://arxiv.org/pdf/1802.04291.pdf), from USC researchers and presented at WWW in 2018, addresses misinformation. In this paper, the researchers used machine learning techniques to “investigate the role and effects of misinformation, using the content produced by Russian Trolls on Twitter as a proxy for misinformation.” They specifically looked at both liberal and conservative media outlets, and particularly focused on “users who re-shared the posts produced on Twitter by the Russian troll accounts publicly disclosed by U.S. Congress investigation” of 2016 election interference. This paper does not focus on actual politicians’ tweets but instead focuses on private citizen retweets; we will focus on the tweets of politicians.

An additional [paper](https://journals.sagepub.com/doi/pdf/10.1177/2158244019827715), from NYU and published in 2019, uses neural networks to classify tweets from Russian accounts as being pro-regime, anti-regime, or neutral. Specifically, the researchers used a “deep feedforward neural network (multilayer perceptron or MLP) that uses a wide range of textual features including words, word pairs, links, mentions, and hashtags to separate four contextually relevant types of trolls: pro-Kremlin, neutral/other, pro-opposition, and pro-Kiev.” The results were “high-confidence predictions for most observations”. This paper focuses on only Russian accounts and does not investigate American politicians, compared to our work which concentrates on Russian-produced tweets that relate to events in United States politics as well as American politicians.

Finally, a [paper](https://arxiv.org/pdf/1802.04289.pdf) from 2018 involving researchers from USC and the Indian Institute of Technology uses neural networks with a contextual long short-term memory (LSTM) architecture that looks at both content and metadata from Twitter accounts to identify trolls among real human users. The authors also used synthetic minority oversampling in order to create a large dataset for training. Our paper will address a similar issue involving classification of troll tweets versus real human user tweets, but with a focus on political speech involving Russian trolls and American Democrat and Republican politicians. Thus, there are a variety of papers that have investigated political tweets, troll tweets, Russian-produced tweets, and differences between troll and real human tweets, but none have specifically looked to distinguish Russian trolls, American Republican politicians, and American Democrat politicians. We hope this paper brings some insight into the area of political tweets and Russian interference in American political discourse on Twitter.

# Methods
The software we used for this project involved Python libraries, namely fastai and Pytorch. We used fastai to create a preliminary text classification model with a average stochastic gradient descent (SGD) weight-dropped long short-term memory (AWD LSTM) architecture, utilizing transfer learning. Then, we focused on using Pytorch for the remainder of the project. Pytorch allowed us to more fully customize a model and get an in-depth understanding of how to create a neural network for a natural language processing application involving political tweets. When we refer to "our model" in this paper, we are referring to this Pytorch model, rather than the fastai one, as that is the model we decided to expend more effort on and were able to customize more. The Pytorch model involved a “bag of words” approach as well as word embeddings to approximate the mathematical meaning of each written tweet. 

We used two datasets from Kaggle, the “Russian Troll Tweets” [dataset](https://www.kaggle.com/fivethirtyeight/russian-troll-tweets) from FiveThirtyEight and the “Democrat Vs. Republican Tweets" [dataset](https://www.kaggle.com/kapastor/democratvsrepublicantweets) from Kyle Pastor. We eliminated irrelevant rows, i.e. Russian Troll tweets not classified as RightTroll or Left Troll. The data was also cleaned to remove non-ascii characters. In the end, the final dataset has a size of around 1 million tweets, with around 622k RightTroll entries, 340k LeftTroll entries, 44k Republican entries, and 42k Democrat entries.  

Once our data was cleaned, we worked on a transfer learning model using a pre-trained AWD LSTM text classification model from fastai. The model was trained on Wikipedia, specifically guessing the next word given all the previous words, following the Universal Language Model Fine-tuning (ULMFiT) approach. We had to reorganize the data into an ImageNet-style organization such that each tweet was contained in a .txt file, and put into folders by class, contained within folder for training and testing data. We first tested our model on a small dataset of 4000 entries, and once convinced we had fixed the bugs, we moved on to using our full dataset. With the full dataset of around 1 million tweets, we fine-tuned the fastai model for 20 epochs, with a base learning rate of .001, with a dropout rate of 25%. 

After using transfer learning, we decided to build a text classification model from scratch using Pytorch, specifically torchtext, to see if we could achieve similar or better results when we were able to have greater control of the hyperparameters and train the model on our specific dataset from the beginning. We first split the data into test and training sets, but kept it in .csv format, rather than in an ImageNet style. After creating a vocabulary using built-in functions from torchtext, and spliting the tweets into batches, we created and trained a TextClassification model, with the vocabulary size, embeddding dimension, and number of classes as some hyperparameters. The architecture of the model is an EmbeddingsBag Layer and then a linear layer. In our loss function, we adjusted the weights to account for the fact that there was much more data on the troll tweets than the poltiician tweets. We trained using 10 epochs, a intial learning rate of 5 and a scheduler gamma of 0.9, and a batch size of 64. 

The tools we used for data analysis were the matplotlib, seaborn, and mlxtend libraries. They allowed us to take our fastai text classification model and Pytorch text classification model and visualize the results using confusion matrices, so we could see where the model succeeded and where it struggled. They also allowed us to compare between the two models. 

# Discussion
Both our fastai model and our Pytorch model yielded significant results. The fastai model provided us with an overall accuracy rate of 87.8%. We determined the per-class accuraccy rates for that model to be the following: 73.2% for Democrat tweets, 73.4% for Republican tweets, 84.7% for LeftTroll tweets, and 91.7% for RightTroll tweets. On the other hand, the Pytorch model provided us with with an overall accuracy of 81.5%, and class specific accuracies of 63.36% for Democrat tweets, 78.89% for Republican tweets, 82.50% for LeftTroll tweets, and 82.42% for RightTroll tweets. 

For the Pytorch model, we tuned the hyperparameters, eventually settling on 10 epochs, a intial learning rate of 5 and a gamma of .9, and a batch size of 64. This was achieved after trying different values for the hyperparameters and adding weights to the loss function to reflect the proportions of the dataset that each class represented. The full details of the hyperparameters tested on the Pytorch model can be seen in Table 1.  

                             Table 1

| Model Framework  | Weighted? | Weights  | Initial LR | Batch Size | Scheduler Gamma | Epochs  | End Accuracy | Dem. Accuracy | Rep. Accuracy | LTroll Accuracy | RTroll Accuracy | End Valid. Cost |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Pytorch  | No  | 1.0, 1.0, 1.0, 1.0 | 5  | 64  | 0.99  | 10  | 83.4%  | 61.35%  | 71.58%  | 76.28%  | 89.64%  | 339.41  |
| Pytorch | Yes  | 1.0, 1.0, 0.125, 0.0714  | 5 | 64  | 0.99 | 10  | 79.9%  | 77.73%  | 70.52%  | 86.29%  | 77.29%  | 530.034  |
| Pytorch | Yes  | 1.0, 1.0, 0.125, 0.0714  | 5  | 64  | 0.9  | 10 | 81.5%  | 63.36%  | 78.89%  | 82.5%  | 82.42%  | 569.205  |

While the fastai model had better overall accuracy than the Pytorch accuracy (87.8% vs. 81.5%), some versions of the Pytorch model had more consistent per-class accuracies than the fastai model, particularly the weighted model. Between the weighted models, while using a gamma value of .9 produced a better overall rate of accuracy, using a gamma rate of .99 produced more consistent per-class accuracies. 

The confusion matrices show that both models have greater difficulty distinguishing between Democrat and Republican tweets and between LeftTroll and RightTroll tweets than between groups of the same partisan slant. In both the confusion matrix for the fastai model (Image 1) and in the confusion matrix for the Pytorch model with gamma of .9 (Image 2), the greatest number of erroneously classified tweets where HERE. This was surprising and heartening to us, as it shows that the Russian propagandists were mostly unsuccessful in emulating U.S. politicians, and that there is perhaps a smaller divide between the two parties, at least in terms of how they structure or word their tweets. 

                             Image 1

<img src="https://user-images.githubusercontent.com/54862430/116016820-a55df900-a5f2-11eb-9717-a9fd86b10af8.png" width="500"> 

                             Image 2
                             
<img src="https://user-images.githubusercontent.com/54862430/116016794-9119fc00-a5f2-11eb-8248-7b19c19a0453.png" width="500">


Compared to others, our work investigates a specific inquiry as to how well a NLP model can distinguish between actual politician tweets and Russian troll tweets, divided by Democrat/Republican and RightTroll/LeftTroll. Other researchers have looked at Russian troll tweets but not in the context of comparing them to American politicians. Because we found that our NLP model can quite accurately determine American politician tweets from Russian troll tweets, our work demonstrates that social media companies may be able to use a model such as ours to flag and/or delete tweets that are likely to come from troll accounts to limit the influence of foreign governments on American politics.

Returning to our our claim, that deep learning can be used to distinguish between American politician tweets and Russian troll tweets by partisan identification, we have demonstrated that it has merit by presenting the results of our model which, with high accuracy, distinguish between different classes of American politicians and trolls. The magnitudes of the accuracies of the models as well as the trends in the confusion matrices show that deep learning can be successful in identifying trolls when compared to politicians on Twitter.

## Conclusion

We have found that a recurrent neural network model can be effectively used to differentiate between Democrat, Republican, LeftTroll, and RightTroll tweets. This builds on the success of previous work and gives hope to the cause of rooting out disinformation from social media, where it can cause political polarization and influence election results.

Further steps we could take would be to look at the troll tweets and compare them to political tweets not written by politicians. As this has somewhat been done before, it would be interesting to see our model, trained on politician tweets as the ‘non-troll’ tweets, would fare compared to other models trained on a broader set of political troll and human tweets. As we've noted, we suspect the model relies upon the greater formality and neatness of the politician tweets to distinguish them from troll tweets, and then uses differences in vocabulary to distinguish between tweets of different political/party affiliation. Since it struggles more with the latter and the layperson who is not a troll is more likely has less neat tweets than a politician, it seems likely that our model would have lower sucess on this new data than it does on the original dataset we've trained it on.  

Other forms of disinformation have also proliferated on Twitter, and on other social media sites like Facebook and Reddit. QAnon is an example of another strain of political disinformation with the potential to wreak havoc. Recently, [research](https://thesoufancenter.org/research/quantifying-the-q-conspiracy-a-data-driven-approach-to-understanding-the-threat-posed-by-qanon/) from the The Soufan Center has suggested that up to 20% of QAnon posts on Facebook in 2020 were created by nation-states, including Russia, China, Iran, and Saudi Arabia. Models like ours should used to weed out that disinformation. There is so much potential in this field to do good, if only the ethical implications are considered.

Our accurate results show the geopolitical importance of employing technological tools like our recurrent neural network models. With such technology, democratic states have a way to fight back against undemocratic states bent on shaping elections to their desire. While of course diplomatic tools will be required, this presents an option that can be used by the private sector unilaterally - it does not require negotiations or agreements. However, because the burden is the private sector, states will need to incentivize social media companies to do so, which presents its own challenges. Nonethenless, such models are a critical tool. 

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
