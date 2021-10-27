# political_ideology_detection

Michigan State University

CSE 842: Natural Language Processing term project

Kevin and Lavanya

## Overview

We will be aggregating a number of datasets containing phrases and their political sentiment (conservative, liberal, neutral). We make sure to distinguish
between the ideology and party (conservative vs. Republican, respectively) and note that we will be studying the ideology.

We won't have millions of samples or endless compute ability.

## NLP/DL methods

There are a number of papers in this field that use a variety of classical and deep methods. Since there are already proven results for older models (Naive Bayes, Logistic 
Regression, RNN, LSTM, CNN), we seek to use modern methods to evaluate this problem. However, we perform an initial baseline test using logistic regression 
as we work up to more complex models.

Some potential methods we will use include RNN/LSTM with Attention, Transformer, and state of the art language models for word embeddings like GloVe, BERT, or ElMo

## Datasets

Below is a sample of datasets that could be used in our experiments. We will likely only use convote but have included three supplementary datasets if we wish
to expand our project.

### [convote](https://www.cs.cornell.edu/home/llee/data/convote.html)

Each speech segment is tagged with metadata such as the party indicator, vote indicator, and tokenized text containing the bill in question.

### [Reddit](https://github.com/jreynolds999/NLP-Reddit-Classification)

1800 sentences labeled as taken from r/Democrats or r/Republicans.

### [Ideological Books Corpus](https://people.cs.umass.edu/~miyyer/ibc/index.html) 

4062 sentences labeled as conservative, liberal, or neutral.

NOTE: need to send email in order to access dataset

### [congressional](https://github.com/jakemsnyder/political-ideology-detection)

A random sample of sentences from the [Congressional Record](https://www.congress.gov/congressional-record) labeled using float in range [-1, 1] that represents
political ideology of speaker. Sentences are also labeled with useful metadata including year, session of congress, speaker, and spoken in HoR or Senate.

Ideology score example: 1 is most conservative member of congress, while -0.235 represents someone who is a more liberal member of congress
