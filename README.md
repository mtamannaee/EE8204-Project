# EE8204-Project
Twitter Sentiment Analysis with Deep Neural Network LSTM

## Overview
The focus of this work is on  classifying the sentiments' poarity behind short natural language texts like Twitter messages. The subject of this project can be considered as a **Natral Language Processing** or **NLP** classification task. 
Its already known for NLP tasks, deep learning methods have outperformed all other methods. Also, in any sequential data like a text, the idea of **Reccurent Neural Network** or **RNN** is proved to be usefull as it can model the input as a sequence of data and learn and represent the output as another sequence or numerical value.
To model a deep sentiment classifier, the sentiments needs to be mapped from a nuerical scores into negative/positive polarities. The model needs to learn to classify the sentiment behind a natural language query and classify its sentiment polarity as negative or pasitive based on query terms. Consequently, a publicly available dataset has been chosen which satisfies such requirement. 

Similar to all NLP tasks, preprocessing has been applied before any other step of training. 

## Dataset
The dataset that has been used for this project is **sentiment140** dataset. It contains **1,600,000 tweets** extracted using the Twitter API. Four types of sentiments has been assigned to these tweets in terms of zero to four score numbers. Based on such score assignment, **score = 0** indicates the most negative polarity score, and the **score = 4** indicates the most positive polarity score. These scores can help us as golden truth to perform sentiment classification on Natural Language texts like tweets. 
 [Kaggle Link](https://www.kaggle.com/kazanova/sentiment140)
 
![](url)
### Dataset Schema
This dataset contains six columns:
 1.  **'Sentiment'**: the polarity of the tweet (0 = negative, 2  = Neutral, 4 = positive)
 2. **'Id'**: The id of the tweet.
 3. **'Date'**: the date of the tweet.
 4. **'Flag'**: If there is no query, then its value is no_query.
 5. **'User'**: the user that tweeted who wrote the tweet.
 6. **'Tweet'** or **'text'**: the text of the tweet.
Thease columns cannot not be considered as features. To train the NN model, we will use the word embedding vectors as features of each tweet. For this aim, we used 

## Code Instruction
This application has been implemented in 5 steps. In the First step the raw dataset's statistics and schema get visualized, in the following next two steps the data gets cleaned and preprocessed. After the tweets are cleaned and stopwords are removed, we train a word embedding model to extract meanigful features. Thease word embeddings then will be feed into the neural network for model training. In the last step keras model gets trained based on the word embedding matrix. The efficiency of the model will furthur get tested based on model's accuracy of classification of test data.

1. https://www.kaggle.com/mahtabtamannaee/sentiment-analysis-lstm/edit
2. https://www.kaggle.com/imvkhandelwal/tensorflow-2-0-rnn-with-glove-vectors
3. https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis/notebook
