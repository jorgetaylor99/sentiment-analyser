# -*- coding: utf-8 -*-
"""
NB sentiment analyser.

Start code.
"""

import argparse
import string
import pandas as pd
from nltk.corpus import stopwords
import nltk

# IMPORTANT, modify this part with your details
USER_ID = "acb20jt"  # your unique student ID


def parse_args():
    parser = argparse.ArgumentParser(description="A Naive Bayes Sentiment \
                    Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words",
                        choices=["all_words", "features"])
    parser.add_argument('-output_files',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix',
                        action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    return args


def load_data(filepath):
    df = pd.read_csv(filepath, sep='\t', header=0)
    return df


# def stoplister():
# def lemmatizer():


def preprocess_data(df):
    punct = string.punctuation.replace('!', '')  # keep exclamation marks!
    stoplist = set(stopwords.words('english'))

    df['Phrase'] = df['Phrase'].str.lower()
    df['Phrase'] = df['Phrase'].str.translate(str.maketrans('', '', punct))
    df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stoplist)]))
    # lemmatisation
    # w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    # lemmatizer = nltk.stem.WordNetLemmatizer()
    # df['Phrase'] = df['Phrase'].apply([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(df['Phrase'])])

    # still some dodgy words getting through!
    return df


def map_sentiment(df):
    df['Sentiment'] = df['Sentiment'].replace(1, 0)
    df['Sentiment'] = df['Sentiment'].replace(2, 1)
    df['Sentiment'] = df['Sentiment'].replace(3, 2)
    df['Sentiment'] = df['Sentiment'].replace(4, 2)
    return df


def compute_priors(df, classes):
    if classes == 3:
        positive_count = df.loc[df['Sentiment'] == 2, 'Sentiment'].count()
        neutral_count = df.loc[df['Sentiment'] == 1, 'Sentiment'].count()
        negative_count = df.loc[df['Sentiment'] == 0, 'Sentiment'].count()
        sum_counts = positive_count + neutral_count + negative_count
        positive_prior = positive_count / sum_counts
        neutral_prior = neutral_count / sum_counts
        negative_prior = negative_count / sum_counts
        return positive_prior, neutral_prior, negative_prior
    else:
        very_positive_count = df.loc[df['Sentiment'] == 4, 'Sentiment'].count()
        positive_count = df.loc[df['Sentiment'] == 3, 'Sentiment'].count()
        neutral_count = df.loc[df['Sentiment'] == 2, 'Sentiment'].count()
        negative_count = df.loc[df['Sentiment'] == 1, 'Sentiment'].count()
        very_negative_count = df.loc[df['Sentiment'] == 0, 'Sentiment'].count()
        sum_counts = very_positive_count + positive_count + neutral_count + negative_count + very_negative_count
        very_positive_prior = very_positive_count / sum_counts
        positive_prior = positive_count / sum_counts
        neutral_prior = neutral_count / sum_counts
        negative_prior = negative_count / sum_counts
        very_negative_prior = very_negative_count / sum_counts
        return very_positive_prior, positive_prior, neutral_prior, negative_prior, very_negative_prior


def main():
    inputs = parse_args()
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    number_classes = inputs.classes
    # accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    # whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
    # whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix

    df_training = preprocess_data(load_data(training))
    df_dev = preprocess_data(load_data(dev))
    df_test = preprocess_data(load_data(test))

    if number_classes == 3:
        df_training = map_sentiment(df_training)
        df_dev = map_sentiment(df_dev)
        print(df_training.head(5))
        positive_prior, neutral_prior, negative_prior = compute_priors(df_training, number_classes)
        print(f'posp: {positive_prior}, neup: {neutral_prior}, negp: {negative_prior}, sum: {positive_prior + neutral_prior + negative_prior}')
    else:
        print('5 classes!')
        very_positive_prior, positive_prior, neutral_prior, negative_prior, very_negative_prior = compute_priors(df_training, number_classes)
        print(f'posp: {positive_prior}, neup: {neutral_prior}, negp: {negative_prior}, sum: {positive_prior + neutral_prior + negative_prior}')

    # You need to change this to return your macro-F1 score for the dev set
    f1_score = 0

    """
    IMPORTANT: your code should return the lines below.
    However, make sure you are also implementing a function to save the class
    predictions on dev and test sets as specified in the assignment handout
    """
    # print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))


if __name__ == "__main__":
    main()
