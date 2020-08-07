# EE8204-Project
Twitter Sentiment Analysis with Deep Neural Network LSTM

## Overview
The focus of this work is on  classifying the sentiment behind short natural language texts like Twitter messages as negative or postive. To perform the classifing task, the sentiments needs to be reported as a nuerical score which can be mapped into negative/positive polarities. The model needs to learn to classify the sentiment behind a natural language query and classify its sentiment polarity as negative or pasitive based on query terms. 

## Dataset
The dataset that has been used for this project is **sentiment140** dataset. It contains **1,600,000 tweets** extracted using the Twitter API. Four types of sentiments has been assigned to these tweets in terms of zero to four score numbers. Based on such score assignment, **score = 0** indicates the most negative polarity score, and the **score = 4** indicates the most positive polarity score. These scores can help us as golden truth to perform sentiment classification on Natural Language texts like tweets. 
 [Kaggle Link](https://www.kaggle.com/kazanova/sentiment140)

## Dataset Schema
This dataset contains six columns:
 1. **'Target'** or **'Sentiment'**: the polarity of the tweet (0 = negative 4 = positive)
 2. **'Id'**: The id of the tweet.
 3. **'Date'**: the date of the tweet.
 4. **'Flag'**: If there is no query, then its value is no_query.
 5. **'User'**: the user that tweeted who wrote the tweet.
 6. **'Tweet'** or **'text'**: the text of the tweet.
The columns **'Id'** , **'Flag'** , and **'User'** wont be beneficial for the purpose of this classification. Thease columns are not be considered as features and get deleted during preprocessing step. However, the  usernames, urls, punctuation marks, extra dots, can provide the sense of relative sentiments.

## Code Instruction
This application has been implemented in 4 steps. In the First step the raw dataset's statistics and schema get visualized, in the following next two steps the data gets cleaned and preprocessed. In the last step **LSTM** model gets trained based on the traing dataset. The efficiency of the model will furthur get tested based on model's accuracy of classification of test data
