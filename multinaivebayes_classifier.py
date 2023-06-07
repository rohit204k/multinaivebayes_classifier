#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
from utils import *


# In[22]:


class MultiNaiveBayes():
    def __init__(self, positive_train_percentage, negative_train_percentage, positive_test_percentage, negative_test_percentage, alpha):
        self.positive_train_percentage = positive_train_percentage
        self.negative_train_percentage = negative_train_percentage

        self.positive_test_percentage = positive_test_percentage
        self.negative_test_percentage = negative_test_percentage
        
        self.alpha = alpha
        
        (self.pos_train, self.neg_train, self.vocab) = load_training_set(self.positive_train_percentage, self.negative_train_percentage)
        (self.pos_test,  self.neg_test) = load_test_set(self.positive_test_percentage, self.negative_test_percentage)
        
        self.pos_train_df = pd.DataFrame(self.pos_train)
        self.neg_train_df = pd.DataFrame(self.neg_train)

        len_pos_train_reviews = len(self.pos_train_df)
        len_neg_train_reviews = len(self.neg_train_df)
        len_train_df = len_pos_train_reviews + len_neg_train_reviews

        self.pos_class_probability = len_pos_train_reviews/len_train_df
        self.neg_class_probability = len_neg_train_reviews/len_train_df


        self.pos_test_df = pd.DataFrame(self.pos_test)
        self.neg_test_df = pd.DataFrame(self.neg_test)

        len_pos_test_reviews = len(self.pos_test_df)
        len_neg_test_reviews = len(self.neg_test_df)
        len_test_df = len_pos_test_reviews + len_neg_test_reviews
        
        # Convert the vocabulary (type : set) to numpy array.

        self.vocab_arr=np.array(list(self.vocab))
        self.vocab_arr.sort()

        self.total_vocabulary = len(self.vocab_arr)

        self.probability_matrix = pd.DataFrame(np.zeros((self.total_vocabulary, 2)),index = self.vocab_arr, columns=['occurance_in_pos_reviews','occurance_in_neg_reviews'])

        self.total_words_in_pos_rev_word_matrix=0
        self.total_words_in_neg_rev_word_matrix=0
        
    def train(self):
        start_time = time.time()
        for row in self.pos_train_df.itertuples():
        #     row_number= row.Index
            row = pd.DataFrame(row)
            words=row[1::].value_counts(dropna=True) # row[0] is index, so excluding it.
            for word, count in words.items():
                word = word[0]  # word is of the form ('movie',) -> converting this to 'movie'
                self.probability_matrix.loc[word]['occurance_in_pos_reviews']+=count
                self.total_words_in_pos_rev_word_matrix+=count

        for row in self.neg_train_df.itertuples():
        #     row_number= row.Index
            row = pd.DataFrame(row)
            words=row[1::].value_counts(dropna=True) # row[0] is index, so excluding it.
            for word, count in words.items():
                word = word[0]                                   
                self.probability_matrix.loc[word]['occurance_in_neg_reviews']+=count
                self.total_words_in_neg_rev_word_matrix+=count
        end_time = time.time()
        print('Training time: {}'.format(end_time-start_time))
        
    def predict(self, instance, alpha):
        instance = instance.dropna()

        temp_p = instance[0].apply(lambda x: self.probability_matrix['occurance_in_pos_reviews'].get(x,0.0))
        pos_probability = self.pos_class_probability*(((temp_p+self.alpha)/(self.total_words_in_pos_rev_word_matrix + self.total_vocabulary*self.alpha)).prod())

        temp_n = instance[0].apply(lambda x: self.probability_matrix['occurance_in_neg_reviews'].get(x,0.0))
        neg_probability = self.neg_class_probability*(((temp_n+self.alpha)/(self.total_words_in_neg_rev_word_matrix + self.total_vocabulary*self.alpha)).prod())

        if pos_probability>=neg_probability:
            return "POSITIVE"
        else:
            return "NEGATIVE"

    def predict_log_trick(self, instance, alpha):
        instance = instance.dropna()

        temp_p = instance[0].apply(lambda x: self.probability_matrix['occurance_in_pos_reviews'].get(x,0.0))
        pos_probability = np.log(self.pos_class_probability) + (np.log((temp_p+self.alpha)/(self.total_words_in_pos_rev_word_matrix + self.total_vocabulary*self.alpha))).sum()

        temp_n = instance[0].apply(lambda x: self.probability_matrix['occurance_in_neg_reviews'].get(x,0.0))
        neg_probability = np.log(self.neg_class_probability) + (np.log((temp_n+self.alpha)/(self.total_words_in_pos_rev_word_matrix + self.total_vocabulary*self.alpha))).sum()

        if pos_probability>=neg_probability:
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
    def test(self, pos_test_set, neg_test_set):
        correct_predictions = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        start_time = time.time()
        for row in pos_test_set.itertuples():

            if self.predict(pd.DataFrame(row)[1::],alpha) == "POSITIVE":
                correct_predictions+=1
                tp+=1
            else:
                fn+=1


        for row in neg_test_set.itertuples():
            if self.predict(pd.DataFrame(row)[1::],alpha) == "NEGATIVE":
                correct_predictions+=1
                tn+=1
            else:
                fp+=1

        end_time = time.time()
        print('\nPrediction (without log trick) time: {}'.format(end_time-start_time))

        print('Correct predictions {} out of {}'.format(correct_predictions,(len(pos_test_set)+len(neg_test_set))))
        print('Accuracy : {}'.format(correct_predictions/(len(pos_test_set)+len(neg_test_set))))
        accuracy = correct_predictions/(len(pos_test_set)+len(neg_test_set))
        confusion = np.array([[tp,fn],[fp,tn]])
        print('Confusion Matrix')
        print(confusion)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print('Precision : {}'.format(precision))
        print('Recall : {}\n'.format(recall))
        
        return accuracy, precision, recall
    
    def test_using_log(self, pos_test_set, neg_test_set):
        correct_predictions = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        start_time = time.time()
        for row in pos_test_set.itertuples():

            if self.predict_log_trick(pd.DataFrame(row)[1::], alpha) == "POSITIVE":
                correct_predictions+=1
                tp+=1
            else:
                fn+=1


        for row in neg_test_set.itertuples():
            if self.predict_log_trick(pd.DataFrame(row)[1::], alpha) == "NEGATIVE":
                correct_predictions+=1
                tn+=1
            else:
                fp+=1

        end_time = time.time()
        print('\nPrediction (with log trick) time: {}'.format(end_time-start_time))

        print('Correct predictions {} out of {}'.format(correct_predictions,(len(pos_test_set)+len(neg_test_set))))
        print('Accuracy : {}'.format(correct_predictions/(len(pos_test_set)+len(neg_test_set))))
        accuracy = correct_predictions/(len(pos_test_set)+len(neg_test_set))
        confusion = np.array([[tp,fn],[fp,tn]])
        print('Confusion Matrix')
        print(confusion)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print('Precision : {}'.format(precision))
        print('Recall : {}\n'.format(recall))
        
        return accuracy, precision, recall


# In[23]:


positive_train_percentage = 0.2
negative_train_percentage = 0.2

positive_test_percentage  = 0.2
negative_test_percentage  = 0.2

alpha = 10#**-5

mnb = MultiNaiveBayes(positive_train_percentage, negative_train_percentage, positive_test_percentage, negative_test_percentage, alpha)


# In[24]:


mnb.train()


# In[25]:


mnb.test(mnb.pos_test_df, mnb.neg_test_df)


# In[28]:


mnb.test_using_log(mnb.pos_test_df, mnb.neg_test_df)


# In[26]:


# Q2 - Laplace smoothing 
# Calculating metrics using various alpha values, and using predict_log_trick to calculate probabilites.
alpha_values = [0.0001,0.001,0.01,0.1,1,10,100,1000]
accuracies = []

for alpha in alpha_values:
    print('Alpha value: {}'.format(alpha))
    mnb.alpha = alpha
    
    accuracy, precision, recall = mnb.test(mnb.pos_test_df, mnb.neg_test_df)
    accuracies.append(accuracy)


# In[27]:


# Plot accuracy vs alpha values plot
import matplotlib.pyplot as plt


plt.plot(alpha_values, accuracies)
  
plt.xlabel('Alpha values')
plt.ylabel('Accuracies')
plt.xscale("log")
  
plt.title('Accuracies for various alpha values')
  
# function to show the plot
plt.show()


# In[29]:


# Q2 - Laplace smoothing 
# Calculating metrics using various alpha values, and using predict_log_trick to calculate probabilites.
alpha_values = [0.0001,0.001,0.01,0.1,1,10,100,1000]
accuracies = []

for alpha in alpha_values:
    print('Alpha value: {}'.format(alpha))
    mnb.alpha = alpha
    
    accuracy, precision, recall = mnb.test_using_log(mnb.pos_test_df, mnb.neg_test_df)
    accuracies.append(accuracy)


# In[30]:


# Plot accuracy vs alpha values plot
import matplotlib.pyplot as plt


plt.plot(alpha_values, accuracies)
  
plt.xlabel('Alpha values')
plt.ylabel('Accuracies')
plt.xscale("log")
  
plt.title('Accuracies for various alpha values')
  
# function to show the plot
plt.show()


# In[ ]:




