# Naive Bayes Sentiment Analyzer

This project is a Python script that uses a Naive Bayes algorithm to perform sentiment analysis on the Rotten Tomatoes Movie Reviews dataset.

## Requirements
The script requires the following Python libraries:
* pandas
* numpy
* nltk
* seaborn
These libraries can be installed using pip.

## Installation

Follow the steps below to install the necessary libraries and setup the environment. 

1. Install pandas, numpy, nltk, and seaborn. Note that you may need to use `pip3` instead of `pip` depending on your Python environment.
    ```bash
    pip install --user -U pandas numpy nltk seaborn
    ```

2. Download the required NLTK packages. Enter Python's interactive mode by typing `python` or `python3` in the terminal, then execute the following commands:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```
    Now exit out of the Python interpreter and return to the command line.

## Usage

The script can be run from the command line with the following syntax:

```bash
python NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes <NUMBER_CLASSES> -features <all_words,features> -output_files -confusion_matrix
```

Replace `<TRAINING_FILE>`, `<DEV_FILE>`, and `<TEST_FILE>` with the paths to your training, development, and testing data files respectively. `<NUMBER_CLASSES>` should be replaced with the number of sentiment classes you want to use. The `-features` flag accepts either `all_words` or `features`, depending on whether you want to include all words in the analysis or just a specific set of features. Use the `-output_files` flag if you want to save the output to files, and the `-confusion_matrix` flag if you want to display a confusion matrix.

## Example

```bash
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 3 -features all_words -output_files -confusion_matrix
```

This command will train the sentiment analyzer using data from `train.tsv`, evaluate it using `dev.tsv`, and make predictions on `test.tsv`. It will use three sentiment classes and include all words in the analysis. It will save the output to files and display a confusion matrix.
