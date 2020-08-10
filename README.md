# EE8204-Project
Twitter Sentiment Analysis with Deep Neural Network LSTM

## Overview
* The focus of this work is on  classifying the sentiments' poarity behind short natural language texts like Twitter messages. The subject of this project can be considered as a **Natral Language Processing** or **NLP** classification task. Its already known for NLP tasks, deep learning methods have outperformed all other methods. Also, in any sequential data like a text, the idea of **Reccurent Neural Network** or **RNN** is proved to be usefull as it can model the input as a sequence of data and learn and represent the output as another sequence or numerical value.
* To model a deep sentiment classifier, the sentiments needs to be mapped from a nuerical scores into **negative/positive** polarities. The model needs to learn to classify the sentiment behind a natural language query and classify its sentiment polarity as negative or pasitive based on query terms. Consequently, a publicly available dataset has been chosen which satisfies such requirement. 

* Similar to all NLP tasks, before any traing step, data preprocessing (Stemming, Lematization, and tokenization) has been performed to get rid of punctuations, hyperlinks, mentions, and meaningless stopwords. The wellknown **NLTK** python library has been used for this purpose. It worthed considering that due to the purpose of sentiment polarity classification, the word "not" can play a role and should not be considered as an stopword, therefore, it should be manualy removed from NLTK stopwords set. 

## Dataset
* The dataset that has been used for this project is **sentiment140** dataset. It contains **1,600,000 tweets** extracted using the Twitter API. This dataset is publicly available at  [Kaggle Link](https://www.kaggle.com/kazanova/sentiment140).
* Four types of sentiments has been assigned to these tweets in terms of zero to four score numbers. Based on such score assignment, **score = 0** indicates the most negative polarity score, and the **score = 4** indicates the most positive polarity score. These scores can help us as golden truth to perform sentiment classification on Natural Language texts like tweets. 


## Dataset Schema
* This dataset contains six columns:

    Row|Sentiment|Id| Date| Flag | User | text |
    -|---------|---|----|------|------|------|
    0|0|1467810369|Mon Apr 06 22:19:45 PDT 2009|NO_QUERY|`_TheSpecialOne_` |@switchfoot http://twitpic.com/2y1zl - Awww, t...|
    1|0|1467810672|Mon Apr 06 22:19:49 PDT 2009|NO_QUERY|scotthamilton|`is upset that he can't update his Facebook by ...` |  
    2|0|1467810917|Mon Apr 06 22:19:53 PDT 2009|NO_QUERY|mattycus|`@Kenichan I dived many times for the ball. Man...`|

  1. **'Sentiment'** or **target**: the polarity of the tweet can be mapped based on this column.
  2. **'Id'**: The id of the tweet.
  3. **'Date'**: the date of the tweet.
  4. **'Flag'**: If there is no query, then its value is no_query.
  5. **'User'**: the user that tweeted who wrote the tweet.
  6. **'Tweet'** or **'text'**: the text of the tweet.
  
* Considering the first column as the output of RNN, the mapping of first column's values are mapped (0 -> negative, 2 -> Neutral, 4 -> positive) as classes. The lable distribution of dataset is shown bellow.

    <img src="https://github.com/mtamannaee/EE8204-Project/blob/master/Figures/Dataset%20Lable%20Dist.PNG" width="550">
 
* Thease columns cannot not be considered as features.To construct a robust model, meaningfull features should represent each query word as the element of an input sequence. Therefore, a **Language Model** wa built to provide a proper representation of a word that can contain contextual meaning with respect to its text. The word embedding vectors can be used as features, in that case each tweet terms' embedding vector coveyes the context of that term in a tweet. 

## Language Model : Word Embedding : Word2Vec

 * The word embedding techniques can be based on global or local informations. Each of these techniques bring a diffrent advatage. Using a global pretrained language model can introduce information from a general point of view by relating out of corpus vocabulary words to the corpus vocabulary, such embedding can make the final model robust. Using a local word embedding models, on the other hand, can bring the precision up as it only gets build based on our local corpus. 
 * Since the purpose is to only focus on this dataset and the testing analysis and results should be reported based on this dataset, a local word embedding model (Word2Vec) was trained and used to construct the embedding layer representing the feature of each text or tweet. The **Word2Vec** word embedding model has been built useing the wellknown **GENSIM** Python library.

## Deep Learning Model : Sequence Model : LSTM

* A mentioned, to model the sequence of words in text, a RNN has been trained with an architecture which includes **Embedding**Layer, **LSTM**, and **Dense** layer.
* Two callbacks gets called at the end of each epoch: First, ReduceLROnPlateau, which updates a Learning Rate at specific epoch. Second, EarlyStopping, which allowes a  performance measure to be specified and monitored. And stoppes the training if it gets triggered.
* This model perdicts an score between 0 and 1. Specifying a THRESHOLD will provide classiffication of the sentiment being positive if the perdicted value is above THRESHOLD. In this project, the THRESHOLD has been set to **"0.5"**, threfore, a tweet gets classified as **POSITIVE** if its perdicted value is above **"0.5"**.

## Model Evaluation :

* In the first step of evaluation, Learning Curve of loss and accuracy of the model on each epoch has been graphed.
  <img src="https://github.com/mtamannaee/EE8204-Project/blob/master/Figures/Accuracy%20Loss.png" width="550">

* Next we can take a look at a Confusion Matric of the models classification.
  <img src="https://github.com/mtamannaee/EE8204-Project/blob/master/Figures/Accuracy%20Loss.png" width="550">

## Model's Classification Results :

* The Precision, recall, and f1-score have been calculated and summerized in this table.


## Code Instruction
1. Download code given in code directory
2. Download the dataset save it in ./data/ directory
3. Install or update all the python libraries.
4. Make sure you are useing Python3.6.x 

