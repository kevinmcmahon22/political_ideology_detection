# political_ideology_detection

Michigan State University

CSE 842: Natural Language Processing term project

Kevin and Lavanya

## Overview

We will be aggregating a number of datasets containing phrases and their political sentiment (conservative, liberal, neutral). 

## NLP/DL methods

There are a number of papers in this field that use a variety of classical and deep methods. Since there are already proven results for older models (Naive Bayes, Logistic 
Regression, RNN, LSTM, CNN), we seek to use modern methods to evaluate this problem with Logistic Regression as a baseline.

Some potential methods we will use include RNN/LSTM with Attention and state of the art language models for word embeddings like GloVe, BERT, or ElMo

## Dataset

### [convote](https://www.cs.cornell.edu/home/llee/data/convote.html)

Each speech segment is tagged with metadata such as the party indicator, vote indicator, and tokenized text containing the bill in question.
