# Installation Guide

From the command line, install pandas and numpy if they are not already installed.  
NOTE: You may have to use pip3 instead of pip!  
`pip install --user -U pandas`  
`pip install --user -U numpy`  
  
Now install nltk which is used for stoplisting, tokenization and lemmatization  
`pip install --user -U nltk`  
Now go to your python command line interpreter:  
`python` or `python3`  
`import nltk`  
`nltk.download('stopwords')`  
`nltk.download('punkt')`  
`nltk.download('wordnet')`  
  
Now exit out of the interpreter and back to the command line.  
Install seaborn for our confusion matrix  
`pip install --user -U seaborn`  
  
The installation should now be complete, you can run the program with:  
python NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes <NUMBER_CLASSES> -features <all_words,features> -output_files -confusion_matrix  
