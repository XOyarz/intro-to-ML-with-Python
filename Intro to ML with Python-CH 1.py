
# coding: utf-8

# In[1]:

from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[2]:

print("Keys ofiris_dataset: \n{}".format(iris_dataset.keys()))


# In[3]:

print(iris_dataset['DESCR'][:193] + "\n")


# In[4]:

print("Target names: {}".format(iris_dataset['target_names']))


# In[5]:

print("features names: {}".format(iris_dataset['feature_names']))


# In[6]:

print("Type of data: {}".format(type(iris_dataset['data'])))


# In[7]:

print("Shape of data: {}".format(iris_dataset['data'].shape))


# In[9]:

print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))


# In[11]:

print("Type of target: {}".format(type(iris_dataset['target'])))


# In[13]:

print("shape of target : {}".format(iris_dataset['target'].shape))


# In[14]:

print("Target:\n{}".format(iris_dataset['target']))


# In[15]:

print("Target:\n{}".format(iris_dataset['target']))


# In[16]:

print("Target:\n{}".format(iris_dataset['target_names']))


# In[17]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[18]:

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[22]:

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', 
                        hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[28]:

import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
get_ipython().system('pip install mglearn')
import mglearn


# In[36]:

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', 
                        hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[37]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[38]:

knn.fit(x_train, y_train)


# In[40]:

x_new = np.array([[5, 2.9, 1, 0.2]])
print("x_new.shape: {}".format(x_new.shape))


# In[41]:

prediction = knn.predict(x_new)
print("prediction: {}".format(prediction))
print("predicted target name: {}".format(iris_dataset['target_names'][prediction]))


# In[42]:

y_pred = knn.predict(x_test)
print("test set predictions:\n {}".format(y_pred))


# In[44]:

print("test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[45]:

print("Test set score: {:.2f}".format(knn.score(x_test, y_test)))


# In[46]:

## Summary code
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

print("Test set score: {:.2f}".format(knn.score(x_test, y_test)))


# In[ ]:



