#!/usr/bin/env python
# coding: utf-8

# In[188]:


import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np


# In[189]:


df = pd.read_csv("C:/Users/Liam/Downloads/ITSC_3162/apple_quality.csv")
df.drop(df.tail(1).index,inplace=True)
df


# In[190]:


df.dtypes


# In[191]:


df.describe


# In[214]:


sns.pairplot(data = df.drop(columns = ["A_id"]))


# In[234]:


from sklearn import tree


# In[253]:


y = df["Quality"]
X = df.drop(columns = ["Quality"])


# In[255]:


from sklearn.model_selection import train_test_split


# In[256]:


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.33)


# In[257]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[258]:


clf = tree.DecisionTreeClassifier()


# In[259]:


clf = clf.fit(X_train, y_train)


# In[260]:


predicted = clf.predict(X_test)


# In[261]:


predicted


# In[262]:


clf.score(X_test,y_test)


# In[263]:


import matplotlib.pyplot as plt


# In[264]:


fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6,6), dpi = 300)
tree.plot_tree(clf, max_depth = 3, feature_names = X.columns, filled = True)
plt.show()


# In[265]:


fi = clf.feature_importances_
fi = pd.Series(data = fi, index = X.columns)
fi.sort_values(ascending = False, inplace = True)


# In[266]:


plt.figure(figsize = (12,6))
chart = sns.barplot(x = fi, y = fi.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation = 45, horizontalalignment = "right")
plt.show


# In[267]:


from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np


# In[268]:


cvs = cross_validate(clf, X, y, cv = 10, return_estimator = True)
cvs["test_score"].mean()


# In[269]:


fi = []
classification_reports = []
for model in cvs ["estimator"]:
    fi.append(list(model.feature_importances_))
fi_avg = np.mean(fi, axis = 0)


# In[270]:


fi_avg


# In[271]:


print(metrics.classification_report(y_test, predicted))


# In[272]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predicted)


# In[339]:


x = df.drop(columns = ["Quality", "Ripeness"])
y = df["Quality"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)


# In[340]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[341]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)


# In[342]:


clf.score(x_test, y_test)


# In[344]:


cvs = cross_validate(clf, x, y, cv = 10, return_estimator = True)
cvs["test_score"].mean()


# In[345]:


cvs["estimator"]


# In[346]:


fi = []
classification_reports = []
for model in cvs ["estimator"]:
    fi.append(list(model.feature_importances_))
fi_avg = np.mean(fi, axis = 0)


# In[347]:


fi_avg


# In[348]:


fi_avg = pd.Series(fi_avg, index = x.columns).sort_values(ascending = False)
plt.figure(figsize = (12,6))
chart = sns.barplot(x = fi_avg, y = fi_avg.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation = 45, horizontalalignment = "right")
plt.show


# In[338]:


predicted = clf.predict(x_test)


# In[284]:


print(metrics.classification_report(y_test, predicted))


# In[287]:


confusion_matrix(y_test, predicted)


# In[ ]:




