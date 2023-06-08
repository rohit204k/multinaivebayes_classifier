# MultiNaiveBayes Classifier - From Scratch
## Simple implementation of Decision Tree Algorithm

MultiNaiveBayes Classifier is supervised learning algorithm which is used in document classication/sentiment analysis.
This is a very simple implementation of MultiNaiveBayes Classification algorithm designed for learning purpose.

Note - The dataset require for this model is not added to the repo, however, it any dataset from the internet can be used as long as the positive and negative documents are separate and present as a list of list of words. Example - The following is the format in which the list of positive and negative documents should be used. Notice that the reviews need to be preprocessed for removing punctuation and cases should be removed. Further preprocessing can also be done to remove remove stop words. Stemming can also be done.

```sh [
  [
    ’movie’, ’ok’, ’kids’, ’gotta’, ’tell’,
  ’ya’, ’scratch’, ’little’, ’squirrel’,
  ’funniest’, ’character’, ’ive’, ’ever’,
  ’seen’, ’makes’, ’movie’, ’hes’,
  ’reason’, ’ive’, ’love’, ’movie’,
  ’congradulations’, ’crew’, ’made’,
  ’laugh’, ’loud’, ’always’
  ],
 [
   ’say’, ’excellent’, ’end’, ’excellent’,’series’, ’never’, ’quite’, ’got’,
    ’exposure’, ’deserved’, ’asia’, ’far’,
    ’best’, ’cop’, ’show’, ’best’,
    ’writing’, ’best’, ’cast’, ’televison’,
    ’ever’, ’end’, ’great’, ’era’, ’sorry’,
    ’see’, ’go’
   ]
]
```

## Features
- The dataset file to run the code is not present in the repo. You can use any of your own dataset to run the code.
- When the number of words in a review is very large, vanishing probabilities might be a problem, to address this, I have implemented the logarithmic trick.
- To test the model, I trained the model using train set and tested it on using different alpha values.
- I have plotted accuracy of model vs alpha values over training and testing data.
- The effect of varying alpha value on accuracy can be studies in the graphs

## Steps to run the code

1. The .ipynb file can be run through jupyter notebook
2. The .py file needs to be run through the following commands.
    * Ensure that all requirements have been installed using
        ```sh
            pip install -r requirements.txt
        ```
    * Run the file using
        ```sh
            python multinaivebayes_classifier.py
        ```
