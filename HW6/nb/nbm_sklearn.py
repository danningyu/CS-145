import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import seaborn as sns; sns.set()
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

# load dataset
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()

# get train and test data
categories = data.target_names
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# set up and train model using sklearn functions
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)

# make predictions on train set to validate the model
predict_train_labels = model.predict(train.data)
train_acc = (train.target==predict_train_labels).mean()
print("Accuracy on training data by sklearn: {}".format(train_acc))

# make predictions on test data
predict_test_labels = model.predict(test.data)

# print test accuracy
test_acc = (test.target==predict_test_labels).mean()
print("Accuracy on test data by sklearn: {}".format(test_acc))

# plot classification matrix
mat = confusion_matrix(test.target, predict_test_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.title('Classification Performace by sklearn')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.savefig('./output/nbm_sklearn.png')