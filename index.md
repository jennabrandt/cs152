# Tweet Classification and Comparison of Russian Propagandists and U.S. Politicians
Jenna Brandt and Erin Puckett

Project literature review linked here: [Literature Review for 2/25/21](https://jennabrandt.github.io/literature-review)

##  Introductory paragraph
We are investigating Twitter usage among Democratic and Republican politicians as well as tweets created by Russian bots. We have created a ML algorithm to effectively categorize the author of a tweet as either a Democrat, a Republican, or a bot.

## Background paragraph
This problem is hard because there are many features of Tweets that could be used to determine whether they were written by bots or real people (message length, frequency, content, time tweeted, number of followers, retweets, etc.). Furthermore, often those behind the bots (especially those in cases like this of state-sponsored disinformation campaigns) closely study the targets in order to appear as similar as possible - there is an active intent to mimic, and that makes tweets hard to tell apart, particularly as some U.S. politicians themselves have embraced disinformation. 

## Transition paragraph
Whereas other existing research has looked at tweet frequency, tweet length, and other metrics to detect bots, we looked at message content. We also compared known bots only to tweets from politicians, the individuals whom those bots are directly emulating, rather than a wider variety of tweets like previous studies have.

## Details paragraph
Technically, we found that it is difficult to assess why our algorithm categorizes certain tweets as it does. Often, bot tweets are shorter than politicians’ tweets, so perhaps instead of looking at the actual language used in the tweets, our algorithm might actually categorize tweets based on length.

## Assessment paragraph
Our algorithm correctly categorizes tweets as Democratic politician, Republican politician, or bot \__% of the time. This means that social media companies could use our algorithm to flag tweets that appear to come from sources who might intend to disrupt American democracy in some way, which would mean that the public (general users of social media) would not be subject to potentially very harmful messaging.

## Sentence on ethics
Freedom of speech is a common concern in today’s use of social media. Social media companies purport to be simply a platform for sharing one’s thoughts, and by classifying tweets as bots, and thus potentially misclassifying real human tweets as bots, we risk dismissing their thoughts as not real and part of a malicious campaign to harm American democracy.
