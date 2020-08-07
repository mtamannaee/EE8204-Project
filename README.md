# EE8204-Project
Twitter Sentiment Analysis with Deep Neural Network

## Overview
The focus of this work is on  classifying the sentiment behind short natural language texts like Twitter messages. To perform the classifing task, the sentiments needs to be reported as a nuerical score which can be mapped into negative/positive polarities.The model needs to learn to classify the sentiment behind a natural language query and classify its sentiment polarity as negative or pasitive based on query terms. 

## Dataset
The dataset that has been used for this project is **sentiment140** dataset. It contains 1,600,000 tweets extracted using the Twitter API. Four types of sentiments has been assigned to these tweets in terms of zero to four score numbers. Based on such score assignment, score 0 indicates the most negative polarity score, and the score 4 indicates the most positive polarity score. These scores can help us as golden truth to perform sentiment classification on Natural Language texts like tweets.
### data schema
This dataset contains six columns:
'target' or 'Sentiment': the polarity of the tweet (0 = negative 4 = positive)
'Id': The id of the tweet.
'Date': the date of the tweet.
'Flag': If there is no query, then this value is NO_QUERY.
'User': the user that tweeted who wrote the tweet.
'Tweet'or 'text': the text of the tweet.

