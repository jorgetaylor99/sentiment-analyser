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
    df = pd.read_csv(filepath, sep='\t', header=0)
    return df

def save_data(df, filename):
    # remove unnecessary columns
    df.drop(['Phrase'], axis=1, inplace=True)
    if 'Sentiment' in df.columns:
        df.drop(['Sentiment'], axis=1, inplace=True)
    # rename predicted to sentiment
    df.rename(columns={'Predicted': 'Sentiment'}, inplace=True)
    # save to file
    df.to_csv('predictions/'+filename, sep='\t', index=False)

def preprocess_data(df, features):
    punct = string.punctuation.replace('!', '')  # keep exclamation marks!
    numbers = string.digits
    stoplist = set(stopwords.words('english'))
    # add custom words to stoplist
    stoplist.update({'nt', 'n', 'lrb', 'nrb', 'rrb', 'le', 'wo', 'pb'})
    lemmatizer = WordNetLemmatizer()

    df['Phrase'] = df['Phrase'].str.lower()
    df['Phrase'] = df['Phrase'].str.translate(str.maketrans('', '', punct))
    df['Phrase'] = df['Phrase'].str.translate(str.maketrans('', '', numbers))
    if features == 'features':
        df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stoplist)]))
    df['Phrase'] = df['Phrase'].apply(lambda x: word_tokenize(x))
    df['Phrase'] = df['Phrase'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])
    return df

def map_sentiment(df):
    df['Sentiment'] = df['Sentiment'].replace(1, 0)
    df['Sentiment'] = df['Sentiment'].replace(2, 1)
    df['Sentiment'] = df['Sentiment'].replace(3, 2)
    df['Sentiment'] = df['Sentiment'].replace(4, 2)
    return df

def compute_priors(df, classes):
    if classes == 3:
        pos_count = df.loc[df['Sentiment'] == 2, 'Sentiment'].count()
        neu_count = df.loc[df['Sentiment'] == 1, 'Sentiment'].count()
        neg_count = df.loc[df['Sentiment'] == 0, 'Sentiment'].count()
        sum_counts = pos_count + neu_count + neg_count
        priors = {2: pos_count / sum_counts, 1: neu_count / sum_counts, 0: neg_count / sum_counts}
    else:
        vpos_count = df.loc[df['Sentiment'] == 4, 'Sentiment'].count()
        pos_count = df.loc[df['Sentiment'] == 3, 'Sentiment'].count()
        neu_count = df.loc[df['Sentiment'] == 2, 'Sentiment'].count()
        neg_count = df.loc[df['Sentiment'] == 1, 'Sentiment'].count()
        vneg_count = df.loc[df['Sentiment'] == 0, 'Sentiment'].count()
        sum_counts = vpos_count + pos_count + neu_count + neg_count + vneg_count
        priors = {4: vpos_count / sum_counts, 3: pos_count / sum_counts, 2: neu_count / sum_counts, 1: neg_count / sum_counts, 0: vneg_count / sum_counts}
    return priors

def compute_likelihoods(df, classes):
    if classes == 3:
        df_positive = df.loc[df['Sentiment'] == 2]
        df_neutral = df.loc[df['Sentiment'] == 1]
        df_negative = df.loc[df['Sentiment'] == 0]
        # print(df_positive['Phrase'].tolist()[:10])
        vocabulary = df_positive['Phrase'].tolist() + df_neutral['Phrase'].tolist() + df_negative['Phrase'].tolist()
        df_list = [(2, df_positive), (1, df_neutral), (0, df_negative)]
        words = {2: [], 1: [], 0: []}
    else:
        df_vpositive = df.loc[df['Sentiment'] == 4]
        df_positive = df.loc[df['Sentiment'] == 3]
        df_neutral = df.loc[df['Sentiment'] == 2]
        df_negative = df.loc[df['Sentiment'] == 1]
        df_vnegative = df.loc[df['Sentiment'] == 0]
        vocabulary = df_vpositive['Phrase'].tolist() + df_positive['Phrase'].tolist() + df_neutral['Phrase'].tolist() + df_negative['Phrase'].tolist() + df_vnegative['Phrase'].tolist()
        df_list = [(4, df_vpositive), (3, df_positive), (2, df_neutral), (1, df_negative), (0, df_vnegative)]
        words = {4: [], 3: [], 2: [], 1: [], 0: []}

    for class_, df in df_list:
        for row in df['Phrase']:
            if classes == 3:
                if class_ == 2:
                    words[2].extend(row)
                elif class_ == 1:
                    words[1].extend(row)
                else:
                    words[0].extend(row)
            else:
                if class_ == 4:
                    words[4].extend(row)
                elif class_ == 3:
                    words[3].extend(row)
                elif class_ == 2:
                    words[2].extend(row)
                elif class_ == 1:
                    words[1].extend(row)
                else:
                    words[0].extend(row)

    wordlist = []
    for row in df['Phrase']:
        wordlist.extend(row)

    vocab = []
    for row in vocabulary:
        for word in row:
            if word not in vocab:
                vocab.append(word)
    print(len(vocab))
    
    likelihoods = {}
    if classes == 5:
        likelihoods[4] = {k: (v + 1) / (len(words[4]) + len(vocab)) for k, v in dict(Counter(words[4])).items()}
        likelihoods[3] = {k: (v + 1) / (len(words[3]) + len(vocab)) for k, v in dict(Counter(words[3])).items()}
    likelihoods[2] = {k: (v + 1) / (len(words[2]) + len(vocab)) for k, v in dict(Counter(words[2])).items()}
    likelihoods[1] = {k: (v + 1) / (len(words[1]) + len(vocab)) for k, v in dict(Counter(words[1])).items()}
    likelihoods[0] = {k: (v + 1) / (len(words[0]) + len(vocab)) for k, v in dict(Counter(words[0])).items()}

    for _class in likelihoods:
        for word in wordlist:
            if word not in likelihoods[_class]:
                likelihoods[_class][word] = 1 / (len(words[_class]) + len(vocab)) 

    return likelihoods

def classifier(df, priors, likelihoods, classes):
    sentiments = []
    for row in df['Phrase']:
        if classes == 3:
            scores = {2: None, 1: None, 0: None}
        else:
            scores = {4: None, 3: None, 2: None, 1: None, 0: None}
        for _class in likelihoods:
            calc = priors[_class]
            for word in row:
                if word in likelihoods[_class]:
                    calc *= likelihoods[_class][word]
            scores[_class] = calc
        best = max(scores, key=scores.get)
        sentiments.append(best)
    df['Predicted'] = sentiments
    return df

def macro_f1(df, classes):
    macro_f1 = 0
    for class_ in range(classes):
        tp = df.loc[(df['Sentiment'] == class_) & (df['Predicted'] == class_), 'Sentiment'].count()
        fp = df.loc[(df['Sentiment'] == class_) & (df['Predicted'] > class_), 'Sentiment'].count()
        fn = df.loc[(df['Sentiment'] == class_) & (df['Predicted'] < class_), 'Sentiment'].count()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        macro_f1 += f1
    macro_f1 /= classes
    return macro_f1

def compute_confusion_matrix(df, classes):
    confusion_matrix = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(classes):
            confusion_matrix[i][j] = df.loc[(df['Sentiment'] == i) & (df['Predicted'] == j), 'Sentiment'].count()
    sns.heatmap(confusion_matrix, annot=True, fmt='.1f', cmap='Blues')
    plt.show()

def main():
    inputs = parse_args()
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    number_classes = inputs.classes
    features = inputs.features
    output_files = inputs.output_files
    confusion_matrix = inputs.confusion_matrix

    # load and preprocess data
    df_training = preprocess_data(load_data(training), features)
    df_dev = preprocess_data(load_data(dev), features)
    df_test = preprocess_data(load_data(test), features)
    
    # map sentiment to 3 classes 
    if number_classes == 3:
        df_training = map_sentiment(df_training)
        df_dev = map_sentiment(df_dev)

    priors = compute_priors(df_training, number_classes)
    likelihoods = compute_likelihoods(df_training, number_classes)
    results_dev = classifier(df_dev, priors, likelihoods, number_classes)
    results_test = classifier(df_test, priors, likelihoods, number_classes)
    f1_score = macro_f1(results_dev, number_classes)

    if confusion_matrix:
        compute_confusion_matrix(results_dev, number_classes)

    if output_files:
        save_data(results_dev, 'dev_predictions_'+str(number_classes)+'classes_acb20jt.tsv')
        save_data(results_test, 'test_predictions_'+str(number_classes)+'classes_acb20jt.tsv')

    # print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()