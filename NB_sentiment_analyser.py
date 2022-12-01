# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import string
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

USER_ID = "acb20jt"  # your unique student ID

def parse_args():
    parser = argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    return args

def load_data(filepath):
    print('LOAD DATA')
    df = pd.read_csv(filepath, sep='\t', header=0)
    return df

def preprocess_data(df):
    print('PREPROCESS DATA')
    punct = string.punctuation.replace('!', '')  # keep exclamation marks!
    stoplist = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    df['Phrase'] = df['Phrase'].str.lower()
    df['Phrase'] = df['Phrase'].str.translate(str.maketrans('', '', punct))
    df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stoplist)]))
    df['Phrase'] = df['Phrase'].apply(lambda x: word_tokenize(x))
    df['Phrase'] = df['Phrase'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])
    return df

def map_sentiment(df):
    print('MAP SENTIMENTS')
    df['Sentiment'] = df['Sentiment'].replace(1, 0)
    df['Sentiment'] = df['Sentiment'].replace(2, 1)
    df['Sentiment'] = df['Sentiment'].replace(3, 2)
    df['Sentiment'] = df['Sentiment'].replace(4, 2)
    return df

def compute_priors(df, classes):
    print("COMPUTE PRIORS")
    if classes == 3:
        pos_count = df.loc[df['Sentiment'] == 2, 'Sentiment'].count()
        neu_count = df.loc[df['Sentiment'] == 1, 'Sentiment'].count()
        neg_count = df.loc[df['Sentiment'] == 0, 'Sentiment'].count()
        sum_counts = pos_count + neu_count + neg_count

        priors = {2: pos_count / sum_counts, 1: neu_count / sum_counts, 0: neg_count / sum_counts}
        return priors
    else:
        vpos_count = df.loc[df['Sentiment'] == 4, 'Sentiment'].count()
        pos_count = df.loc[df['Sentiment'] == 3, 'Sentiment'].count()
        neu_count = df.loc[df['Sentiment'] == 2, 'Sentiment'].count()
        neg_count = df.loc[df['Sentiment'] == 1, 'Sentiment'].count()
        vneg_count = df.loc[df['Sentiment'] == 0, 'Sentiment'].count()
        sum_counts = vpos_count + pos_count + neu_count + neg_count + vneg_count

        priors = {4: vpos_count / sum_counts, 3: pos_count / sum_counts, 2: neu_count / sum_counts, 1: neg_count / sum_counts, 0: vneg_count / sum_counts}
        return priors

def compute_likelihoods(df):
    print("COMPUTE LIKELIHOODS")

    df_positive = df.loc[df['Sentiment'] == 2]
    df_neutral = df.loc[df['Sentiment'] == 1]
    df_negative = df.loc[df['Sentiment'] == 0]
    vocabulary = df_positive['Phrase'].tolist() + df_neutral['Phrase'].tolist() + df_negative['Phrase'].tolist()
    print(len(vocabulary))
    df_list = [(2, df_positive), (1, df_neutral), (0, df_negative)]

    words = {2: [], 1: [], 0: []}
    for class_, df in df_list:
        for row in df['Phrase']:
            if class_ == 2:
                words[2].extend(row)
            elif class_ == 1:
                words[1].extend(row)
            else:
                words[0].extend(row)

    wordlist = []
    for row in df['Phrase']:
        wordlist.extend(row)

    likelihoods = {}
    likelihoods[2] = {k: (v + 1) / (len(words[2]) + len(vocabulary)) for k, v in dict(Counter(words[2])).items()}
    likelihoods[1] = {k: (v + 1) / (len(words[1]) + len(vocabulary)) for k, v in dict(Counter(words[1])).items()}
    likelihoods[0] = {k: (v + 1) / (len(words[0]) + len(vocabulary)) for k, v in dict(Counter(words[0])).items()}

    for _class in likelihoods:
        for word in wordlist:
            if word not in likelihoods[_class]:
                # likelihoods[_class][word] = 0
                likelihoods[_class][word] = 1 / (len(words[_class]) + len(vocabulary)) 

    print(likelihoods)
    return likelihoods

def classifier(df, priors, likelihoods, classes):
    print("CLASSIFIER")

    # a function which takes a sentence and returns the most likely class
    # for each word in the sentence, multiply the likelihood of that word given the class by the prior of the class
    sentiments = []
    for row in df['Phrase']:
        print(row)
        scores = {2: None, 1: None, 0: None}
        for _class in likelihoods:
            calc = priors[_class]
            for word in row:
                if word in likelihoods[_class]:
                    calc *= likelihoods[_class][word]
                    # print(_class, word, calc)
            scores[_class] = calc
        # print(scores) 
        best = max(scores, key=scores.get)
        sentiments.append(best)
    df['Predicted'] = sentiments

    print(df.to_string())
    return df

def macro_f1(df, classes):
    print("MACRO F1")

    for class_ in range(classes):
        print(class_, '------------------------------------------')
        tp = df.loc[(df['Sentiment'] == class_) & (df['Predicted'] == class_), 'Sentiment'].count()
        tn = df.loc[(df['Sentiment'] == 1) & (df['Predicted'] == 1), 'Sentiment'].count()
        fp = df.loc[(df['Sentiment'] == 1) & (df['Predicted'] == 2), 'Sentiment'].count()
        fn = df.loc[(df['Sentiment'] == 2) & (df['Predicted'] == 1), 'Sentiment'].count()

        print(tp, tn, fp, fn)
        print(sum([tp, tn, fp, fn]))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        print("F1: ", f1)

def compute_confusion_matrix(df, classes):
    print("CONFUSION MATRIX")

    confusion_matrix = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(classes):
            confusion_matrix[i][j] = df.loc[(df['Sentiment'] == i) & (df['Predicted'] == j), 'Sentiment'].count()
    print(confusion_matrix)
    print(confusion_matrix.sum())

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
        priors = compute_priors(df_training, number_classes)
    else:
        priors = compute_priors(df_training, number_classes)

    likelihoods = compute_likelihoods(df_training)
    results = classifier(df_dev, priors, likelihoods, number_classes)
    macro_f1(results, number_classes)
    # You need to change this to return your macro-F1 score for the dev set
    f1_score = 0

    if confusion_matrix:
        compute_confusion_matrix(results, number_classes)

    """
    IMPORTANT: your code should return the lines below.
    However, make sure you are also implementing a function to save the class
    predictions on dev and test sets as specified in the assignment handout
    """
    # print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()