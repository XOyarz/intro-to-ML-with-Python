
# coding: utf-8

# In[7]:

import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
get_ipython().system('pip install mglearn')
import mglearn


# In[3]:

# generate dataset
x, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(x[:, 0], x[:, 1],  y)
plt.legend(["class 0", "class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("second feature")
print("x.shape: {}".format(x.shape))


# In[4]:

x, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(x, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")


# In[5]:

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))


# In[6]:

print("Shape of cancer data: {}".format(cancer.data.shape))


# In[7]:

print("Sample counts per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))


# In[8]:

print("Feature names:\n{}".format(cancer.feature_names))


# In[9]:

from sklearn.datasets import load_boston
boston = load_boston()
print("Data Shape: {}".format(boston.data.shape))


# In[10]:

x, y = mglearn.datasets.load_extended_boston()
print("x.shape: {}".format(x.shape))


# In[11]:

mglearn.plots.plot_knn_classification(n_neighbors=1)


# In[12]:

mglearn.plots.plot_knn_classification(n_neighbors=3)


# In[13]:

from sklearn.model_selection import train_test_split
x, y = mglearn.datasets.make_forge()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[14]:

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


# In[15]:

clf.fit(x_train, y_train)


# In[16]:

print("Test set predictions: {}".format(clf.predict(x_test)))


# In[17]:

print("Test set accuracy: {:.2f}".format(clf.score(x_test, y_test)))


# In[20]:

# decision boundary graph
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x, y)
    mglearn.plots.plot_2d_separator(clf, x, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(x[:, 0], x[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)


# In[21]:

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# try N_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train, y_train)
    # record training accuracy
    training_accuracy.append(clf.score(x_train, y_train))
    # record_generalization accuracy
    test_accuracy.append(clf.score(x_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[22]:

## k-nearest neighbors for regression
from sklearn.neighbors import KNeighborsRegressor
x, y = mglearn.datasets.make_wave(n_samples=40)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(x_train, y_train)


# In[23]:

print("Test set predictions:\n{}".format(reg.predict(x_test)))


# In[25]:

print("Test set R^2: {:.2f}".format(reg.score(x_test, y_test)))


# In[4]:

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make preds using 1, 3, 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(x_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(x_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(x_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
    n_neighbors, reg.score(x_train, y_train), reg.score(x_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")


# In[5]:

mglearn.plots.plot_linear_regression_wave()


# In[10]:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x, y = mglearn.datasets.make_wave(n_samples=60)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

lr = LinearRegression().fit(x_train, y_train)


# In[12]:

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))


# In[13]:

print("Training set score: {:.2f}".format(lr.score(x_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(x_test, y_test)))


# In[14]:

# linear regression on complex dataset (many features)
x, y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lr = LinearRegression().fit(x_train, y_train)


# In[15]:

print("Training set score: {:.2f}".format(lr.score(x_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(x_test, y_test)))


# In[16]:

from sklearn.linear_model import Ridge

ridge = Ridge().fit(x_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(x_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(x_test, y_test)))


# In[17]:

# playing around with alpha
ridge10 = Ridge(alpha=10).fit(x_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(x_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(x_test, y_test)))


# In[19]:

ridge01 = Ridge(alpha=0.1).fit(x_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(x_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(x_test, y_test)))


# In[21]:

from sklearn.linear_model import Lasso

lasso = Lasso().fit(x_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(x_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))


# In[23]:

## decrease alpha, increase iterations
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(x_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(x_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))


# In[29]:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
get_ipython().magic('matplotlib inline')
X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()


# In[28]:

mglearn.plots.plot_linear_svc_regularization()


# In[30]:

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(x_train, y_train)
print("Training test score: {:.3f}".format(logreg.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(x_test, y_test)))


# In[32]:

logreg100 = LogisticRegression(C=100).fit(x_train, y_train)
print("Training test score: {:.3f}".format(logreg100.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(x_test, y_test)))


# In[33]:

logreg001 = LogisticRegression(C=0.01).fit(x_train, y_train)
print("Training test score: {:.3f}".format(logreg001.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(x_test, y_test)))


# In[34]:

# linear models for multiclass classification
from sklearn.datasets import make_blobs

x, y = make_blobs(random_state=42)
mglearn.discrete_scatter(x[:, 0], x[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["class 0", "class 1", "class 2"])


# In[35]:

linear_svm = LinearSVC().fit(x, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)


# In[36]:

# Naive Bayes Classifier
x = np.array([[0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])


# In[37]:

counts = {}
for label in np.unique(y):
    # iterate over each class
    # count (sum) entries of 1 per feature
    counts[label] = x[y == label].sum(axis=0)
print("Feature counts:\n{})".format(counts))


# In[38]:

mglearn.plots.plot_animal_tree()


# In[39]:

from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))


# In[41]:

# pre-pruning to avoid overfitting
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))


# In[42]:

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], 
               feature_names=cancer.feature_names, impurity=False, filled=True)


# In[44]:

get_ipython().system('pip install graphviz')
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[45]:

print("Feature importances:\n{}".format(tree.feature_importances_))


# In[46]:

tree = mglearn.plots.plot_tree_not_monotone()
display(tree)


# In[47]:

import pandas as pd
ram_prices = pd.read_csv("data/ram_price.csv")

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("year")
plt.ylabel("Price in $/Mbyte")


# In[ ]:

## ensembles of decision trees pg 83

