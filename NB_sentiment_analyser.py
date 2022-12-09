# -*- coding: utf-8 -*-
import math
import string
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt

USER_ID = "acb20jt"

def parse_args():
    '''
    Parse command line arguments, set defaults where no argument is provided.
    Return:
        args: the parsed arguments
    '''
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
    '''
    Load data from file as a pandas dataframe.
    Args:
        filepath: the path to the file
    Return:
        df: the loaded dataframe
    '''
    df = pd.read_csv(filepath, sep='\t', header=0)

    return df

def save_data(df, filename):
    '''
    Save data to file as a tab-separated file.
    Remove unnecessary columns and renames the predicted sentiment column to 'Sentiment'.
    Args:
        df: the dataframe to save
        filename: the name of the file to save to
    '''
    df.drop(['Phrase'], axis=1, inplace=True)

    if 'Sentiment' in df.columns:
        df.drop(['Sentiment'], axis=1, inplace=True)

    df.rename(columns={'Predicted': 'Sentiment'}, inplace=True)
    df.to_csv('predictions/'+filename, sep='\t', index=False)

def preprocess_data(df, features):
    '''
    Preprocess the data by removing punctuation and numbers
    Also lowercase every word and lemmatize.
    If specific features are specified, remove stopwords.
    Args:
        df: the dataframe to preprocess
        features: the features to use
    Return:
        df: the preprocessed dataframe
    '''
    punct = string.punctuation.replace('!', '')  # keep exclamation marks!
    stoplist = set(stopwords.words('english'))
    stoplist.update({'nt', 'n', 'lrb', 'nrb', 'rrb', 'le', 'wo', 'pb'})  # add some more stopwords
    lemmatizer = WordNetLemmatizer()

    df['Phrase'] = df['Phrase'].str.lower()
    df['Phrase'] = df['Phrase'].str.translate(str.maketrans('', '', punct))
    if features == 'features':
        df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stoplist)]))
    df['Phrase'] = df['Phrase'].apply(lambda x: word_tokenize(x))
    df['Phrase'] = df['Phrase'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])
    # if features == 'features':
    #     df['Phrase'] = df['Phrase'].apply(nltk.pos_tag)
    #     df['Phrase'] = df['Phrase'].apply(lambda x: [word for word, tag in x if tag in ['JJ', 'JJR', 'JJS']])

    return df

def map_sentiment(df):
    '''
    If 3 classes are specified, map the sentiment values to remove very positive and very negative.
    Adjust the sentiment values to be 0, 1, 2.
    Args:
        df: the dataframe to map
    Return:
        df: the mapped dataframe
    '''
    df['Sentiment'] = df['Sentiment'].replace(1, 0)
    df['Sentiment'] = df['Sentiment'].replace(2, 1)
    df['Sentiment'] = df['Sentiment'].replace(3, 2)
    df['Sentiment'] = df['Sentiment'].replace(4, 2)

    return df

def compute_priors(df, classes):
    '''
    Compute the priors for the specified number of classes.
    Args:
        df: the dataframe to compute the priors from
        classes: the number of classes
    Return:
        priors: a dictionary with the priors for each class
    '''
    priors = {}
    sum_counts = df['Sentiment'].count()

    for _class in range(classes):
        count = df.loc[df['Sentiment'] == _class, 'Sentiment'].count()
        priors[_class] = count / sum_counts

    return priors

def compute_likelihoods(df, classes):
    '''
    Compute the likelihoods for the specified number of classes.
    Args:
        df: the dataframe to compute the likelihoods from
        classes: the number of classes
    Return:
        likelihoods: a dictionary with the likelihoods for each class
    '''
    # calculate words where key is class and value is list of words from the reviews of that class
    words = {key: [] for key in range(classes)}
    for class_ in range(classes):
        for row in df.loc[df['Sentiment'] == class_, 'Phrase']:
            words[class_].extend(row)
                
    # calcalate vocabulary which is a list of all unique words from all reviews
    vocabulary = df['Phrase'].tolist()
    vocab = set()
    for row in vocabulary:
        for word in row:
            vocab.add(word)
    
    # calculate likelihoods for each word in vocabulary for each class
    likelihoods = {}
    for class_ in range(classes):
        likelihoods[class_] = {k: (v + 1) / (len(words[class_]) + len(vocab)) for k, v in dict(Counter(words[class_])).items()}

    # add 1 laplace smoothing for words that are not in the vocabulary
    for _class in likelihoods:
        for word in vocab:
            if word not in likelihoods[_class]:
                likelihoods[_class][word] = 1 / (len(words[_class]) + len(vocab)) 

    return likelihoods

def classifier(df, priors, likelihoods, classes):
    '''
    Classify the test data using the priors and likelihoods. 
    Args:
        df: the dataframe to classify
        priors: the priors for each class
        likelihoods: the likelihoods for each class
        classes: the number of classes
    Return:
        df: the classified dataframe
    '''
    sentiments = []

    for row in df['Phrase']:
        scores = {key: None for key in range(classes)}
        for _class in likelihoods:
            calc = priors[_class]
            for word in row:
                if word in likelihoods[_class]:
                    # calc *= likelihoods[_class][word]
                    calc += math.log(likelihoods[_class][word])
            scores[_class] = calc
        highest_score = max(scores, key=scores.get)
        sentiments.append(highest_score)
    df['Predicted'] = sentiments

    return df

def macro_f1(df, classes):
    '''
    Compute the macro f1 score for the specified number of classes.
    Args:
        df: the dataframe to compute the macro f1 score from
        classes: the number of classes
    Return:
        macro_f1: the macro f1 score
    '''
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
    '''
    Compute the confusion matrix for the specified number of classes.
    Uses the seaborn library to plot the confusion matrix.
    Args:
        df: the dataframe to compute the confusion matrix from
        classes: the number of classes
    '''
    confusion_matrix = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(classes):
            confusion_matrix[i][j] = df.loc[(df['Sentiment'] == i) & (df['Predicted'] == j), 'Sentiment'].count()
    cm = sns.heatmap(confusion_matrix, annot=True, fmt='.1f', cmap='Blues')
    cm.set(xlabel='Predicted', ylabel='Actual')
    plt.show()

def main():
    '''
    Main function to run the program.
    1. Parses the command line arguments
    2. Preprocesses the data
    3. Maps sentiments if necessary
    4. Computes the priors and likelihoods
    5. Classifies the data
    6. Computes the macro f1 score
    7. Computes the confusion matrix if specified
    8. Writes the results to the specified output files if specified
    '''
    inputs = parse_args()
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    number_classes = inputs.classes
    features = inputs.features
    output_files = inputs.output_files
    confusion_matrix = inputs.confusion_matrix

    df_training = preprocess_data(load_data(training), features)
    df_dev = preprocess_data(load_data(dev), features)
    df_test = preprocess_data(load_data(test), features)
    
    if number_classes == 3:
        df_training = map_sentiment(df_training)
        df_dev = map_sentiment(df_dev)

    priors = compute_priors(df_training, number_classes)
    likelihoods = compute_likelihoods(df_training, number_classes)
    results_dev = classifier(df_dev, priors, likelihoods, number_classes)
    results_test = classifier(df_test, priors, likelihoods, number_classes)
    f1_score = macro_f1(results_dev, number_classes)

    # note: the confusion matrix must be closed before the output files are saved
    if confusion_matrix:
        compute_confusion_matrix(results_dev, number_classes)

    if output_files:
        save_data(results_dev, 'dev_predictions_'+str(number_classes)+'classes_acb20jt.tsv')
        save_data(results_test, 'test_predictions_'+str(number_classes)+'classes_acb20jt.tsv')

    print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()