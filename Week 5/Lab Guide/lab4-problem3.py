import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

'''
Step 1: Read file using pandas from a URL and prepare our dataset
'''
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms_dataset = pd.read_table(url, header=None, names=['label', 'message'])

# This is a tip !
# We can examine the class distribution using method 'value_counts()' on a certain feature.
# For example, we want to examine a distribution over labels.
print('Class distribution over labels:', '\n', sms_dataset.label.value_counts(), '\n')

# Augment sms_dataset with a numerical value corresponding each label
sms_dataset['label_num'] = sms_dataset.label.map({'ham':0, 'spam':1})
print('Let\'s see what we got for the sms_dataset:', '\n', sms_dataset, '\n')

# It's time to construct a feature matrix X and a target vector y
X = sms_dataset.message
y = sms_dataset.label_num

'''
Step 2: Constructs a vocabulary from the dataset. This step is called 'vectorizing the dataset'. 
To vectorizing in sklearn, we instantiate CountVectorizer and fit with the dataset. 
This vocabulary will be used later to create a document-term matrix !
'''

vectorizer = CountVectorizer() # instantiate the vectorizer

# Read the following document by yourself and figure out which method can be used to vectorize the training examples
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# Put your code here

# This can be rewritten as vectorizer.fit_transform(X)
vectorizer.fit(X)
X_document_term_matrix = vectorizer.transform(X)

'''
Step 3: Instantiate a classifier. Noted that the multinomial naive bayes classifier is suitable for 
classifying with discrete features (e.g. word counts for text classification). The multinomial distribution 
normally requires integer feature counts. However, in practice, fractional counts may also work. 
'''

multinomialNaiveBayes = MultinomialNB()

# To train the model, we use method 'fit'
# See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
multinomialNaiveBayes.fit(X_document_term_matrix, y)

'''
Step 4: Measure accuracy of our model. 
'''

predicted_y = multinomialNaiveBayes.predict(X_document_term_matrix)
print(metrics.accuracy_score(y, predicted_y))
