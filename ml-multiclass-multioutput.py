#!/usr/bin/env python
# coding: utf-8

# # Multiclass classification

# ### Collecting data

# - [x] Explore [sklearn](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification) for **Multiclass classification**, **Multilabel classification** and **Multioutput Regression**. The main components of the library are shown in the figure below.
# ![multi_org_chart](https://scikit-learn.org/stable/_images/multi_org_chart.png)

# - [x] Find data that can solve the **Multiclass classification** problem (classification with more than two classes). As a last resort, convert data intended for another task.

# In[1]:


import pandas as pd
pd.set_option("display.max.columns", None)


# In[2]:


df = pd.read_csv("./data/Chicago_Crime_Final_Data.csv")
df


# Since the dataset was already cleaned [here](https://github.com/Mark-Rozenberg/Crime-And-Climate/blob/main/Chicago_Crime_PreProcess.ipynb), I am skipping the data cleaning steps.

# - [x] If necessary, perform useful data transformations (for example, transform categorical features into quantitative ones), remove unnecessary features, create new ones (**Feature Engineering**).

# ### Feature engineering

# In[3]:


df.columns


# According to the data description [here](https://github.com/Mark-Rozenberg/Crime-And-Climate/blob/main/README.md), the column names are changed for better understandibility.

# In[4]:


df.columns = ['crime_type', 'location_desc', 'police_area', 'city_district', 'date', 'hour',
       'police_dist', 'precipitation', 'snowfall', 'snow_depth', 'temp_max', 'temp_min', 'wind_dir', 'wind_speed',
       'fog', 'fog_heavy', 'thunder', 'ice_pellet', 'glaze_rime', 'smoke_haze', 'snow_drift', 'day_week',
       'month', 'holiday']
df


# In[5]:


frequent_crimes = ['BATTERY','THEFT','CRIMINAL DAMAGE']
df = df[df['crime_type'].isin(frequent_crimes)]
df


# Since the volume of the original data is too big for our purpose, we'll be sampling 1000 records randomly.

# In[6]:


df = df.sample(n=1000)


# In[ ]:





# In[7]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
crime_types = le.fit_transform(df["crime_type"])
df["crime_type_cat"] = crime_types
df


# In[8]:


locations = le.fit_transform(df["location_desc"])
locations
df["location_desc_cat"] = locations
df


# ### EDA

# - [x] Perform exploratory analysis (**EDA**), use visualization, draw conclusions that may be useful in further solving the problem.

# In[9]:


df.info()


# In[10]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

correlation = df.corr()

mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(180, 20, as_cmap=True)
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()


# In[11]:


pd.DataFrame(df.apply(lambda col: len(col.unique())),columns=["Unique Values Count"])


# In[12]:


df.describe().T


# In[13]:


df["crime_type_cat"].plot.hist()


# In[14]:


df.groupby("crime_type_cat").mean()


# In[15]:


pd.crosstab(df.crime_type,df.month).plot(kind="bar", figsize=(20, 5))
plt.title("Crime Type Frequency for Month")
plt.xlabel("Crime Type")
plt.ylabel("Month")
plt.show()


# In[16]:


pd.crosstab(df.crime_type,df.day_week).plot(kind="bar", figsize=(20, 5))
plt.title("Crime Type Frequency for Day of Week")
plt.xlabel("Crime Type")
plt.ylabel("Day of Week")
plt.show()


# In[17]:


pd.crosstab(df.crime_type,df.holiday).plot(kind="bar", figsize=(20, 5))
plt.title("Crime Type Frequency for Holiday")
plt.xlabel("Crime Type")
plt.ylabel("Holiday")
plt.show()


# In[18]:


pd.crosstab(df.crime_type,df.fog).plot(kind="bar", figsize=(20, 5))
plt.title("Crime Type Frequency for Fog")
plt.xlabel("Crime Type")
plt.ylabel("Fog")
plt.show()


# In[19]:


pd.crosstab(df.crime_type,df.thunder).plot(kind="bar", figsize=(20, 5))
plt.title("Crime Type Frequency for Thunder")
plt.xlabel("Crime Type")
plt.ylabel("Thunder")
plt.show()


# In[20]:


pd.crosstab(df.crime_type,df.smoke_haze).plot(kind="bar", figsize=(20, 5))
plt.title("Crime Type Frequency for Smoke/Haze")
plt.xlabel("Crime Type")
plt.ylabel("Smoke/Haze")
plt.show()


# In[21]:


plt.scatter(x=df.month[df.temp_max>=0], y=df.wind_speed[(df.temp_max>=0)])
plt.scatter(x=df.month[df.temp_max<0], y=df.wind_speed[(df.temp_max<0)])
plt.legend(["Plus", "Minus"])
plt.xlabel("Month")
plt.ylabel("Wind speed")
plt.show()


# In[22]:


df = df.drop(columns=["crime_type", "location_desc", "date", "fog", "fog_heavy", "thunder", "ice_pellet", "glaze_rime", "smoke_haze", "snow_drift", "holiday"])


# We will try training without the dummy features from here on.

# In[23]:


sns.pairplot(df)


# ### Training models

# - [x] Using the **OneVsRest**, **OneVsOne** and **OutputCode** strategies, solve the **Multiclass classification** problem for each of the basic classification algorithms passed (**logistic regression, svm, knn, naive bayes, decision tree**). When training, use **hyperparameter fitting**, **cross-validation** and, if necessary, **data scaling**, to achieve the best prediction quality.
# - [x] Measure the training time of each model for each strategy.
# - [x] To assess the quality of models, use the **AUC-ROC** metric.

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

X, y = df.drop(columns=["crime_type_cat"]), df["crime_type_cat"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

training_results = []


# #### Logistic regression

# In[25]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

parameters = {"penalty" : ["l1", "l2"], "C": np.logspace(-3,3,7)}
cross_validation = StratifiedKFold(n_splits=10,shuffle=True,random_state=123)
gs_logregr = GridSearchCV(LogisticRegression(), parameters, scoring="roc_auc", cv=cross_validation)


# In[26]:


# OneVsRestClassifier

start = time.time()
ovr_logregr = OneVsRestClassifier(gs_logregr)
ovr_logregr.fit(X_train_scaled, y_train)
y_pred = ovr_logregr.predict(X_test_scaled)
ovr_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovr_score)

training_results.append({"model": "logistic-regression", "strategy": "one-vs-rest", "time": training_time, "accuracy": ovr_score})


# In[27]:


# OneVsOneClassifier

start = time.time()
ovo_logregr = OneVsOneClassifier(gs_logregr)
ovo_logregr.fit(X_train_scaled, y_train)
y_pred = ovo_logregr.predict(X_test_scaled)
ovo_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovo_score)

training_results.append({"model": "logistic-regression", "strategy": "one-vs-one", "time": training_time, "accuracy": ovo_score})


# In[28]:


# OutputCodeClassifier

start = time.time()
oc_logregr = OutputCodeClassifier(gs_logregr)
oc_logregr.fit(X_train_scaled, y_train)
y_pred = oc_logregr.predict(X_test_scaled)
oc_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", oc_score)

training_results.append({"model": "logistic-regression", "strategy": "output-code", "time": training_time, "accuracy": oc_score})


# #### Support vector machine

# In[29]:


from sklearn.svm import SVC

parameters = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001],
              "kernel": ["rbf"]}
gs_svm = GridSearchCV(SVC(), parameters, scoring="roc_auc", cv=10)


# In[30]:


# OneVsRestClassifier

start = time.time()
ovr_svm = OneVsRestClassifier(gs_svm)
ovr_svm.fit(X_train_scaled, y_train)
y_pred = ovr_svm.predict(X_test_scaled)
ovr_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovr_score)

training_results.append({"model": "svm", "strategy": "one-vs-rest", "time": training_time, "accuracy": ovr_score})


# In[31]:


# OneVsOneClassifier

start = time.time()
ovo_svm = OneVsOneClassifier(gs_svm)
ovo_svm.fit(X_train_scaled, y_train)
y_pred = ovo_svm.predict(X_test_scaled)
ovo_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovo_score)

training_results.append({"model": "svm", "strategy": "one-vs-one", "time": training_time, "accuracy": ovo_score})


# In[32]:


# OutputCodeClassifier

start = time.time()
oc_svm = OutputCodeClassifier(gs_svm)
oc_svm.fit(X_train_scaled, y_train)
y_pred = oc_svm.predict(X_test_scaled)
oc_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", oc_score)

training_results.append({"model": "svm", "strategy": "output-code", "time": training_time, "accuracy": oc_score})


# #### K-nearest neighbours

# In[33]:


from sklearn.neighbors import KNeighborsClassifier

parameters = {"n_neighbors": [3, 5, 11, 19]}
gs_knn = GridSearchCV(KNeighborsClassifier(), parameters, scoring="roc_auc", cv=10)


# In[34]:


# OneVsRestClassifier

start = time.time()
ovr_knn = OneVsRestClassifier(gs_knn)
ovr_knn.fit(X_train_scaled, y_train)
y_pred = ovr_knn.predict(X_test_scaled)
ovr_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovr_score)

training_results.append({"model": "knn", "strategy": "one-vs-rest", "time": training_time, "accuracy": ovr_score})


# In[35]:


# OneVsOneClassifier

start = time.time()
ovo_knn = OneVsOneClassifier(gs_knn)
ovo_knn.fit(X_train_scaled, y_train)
y_pred = ovo_knn.predict(X_test_scaled)
ovo_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovo_score)

training_results.append({"model": "knn", "strategy": "one-vs-one", "time": training_time, "accuracy": ovo_score})


# In[36]:


# OutputCodeClassifier

start = time.time()
oc_knn = OutputCodeClassifier(gs_knn)
oc_knn.fit(X_train_scaled, y_train)
y_pred = oc_knn.predict(X_test_scaled)
oc_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", oc_score)

training_results.append({"model": "knn", "strategy": "output-code", "time": training_time, "accuracy": oc_score})


# #### Naive Bayes

# In[37]:


from sklearn.naive_bayes import GaussianNB

parameters = {"var_smoothing": np.logspace(0,-9, num=100)}
gs_nb = GridSearchCV(GaussianNB(), parameters, scoring="roc_auc", cv=10)


# In[38]:


# OneVsRestClassifier

start = time.time()
ovr_nb = OneVsRestClassifier(gs_nb)
ovr_nb.fit(X_train_scaled, y_train)
y_pred = ovr_nb.predict(X_test_scaled)
ovr_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovr_score)

training_results.append({"model": "naive-bayes", "strategy": "one-vs-rest", "time": training_time, "accuracy": ovr_score})


# In[39]:


# OneVsOneClassifier

start = time.time()
ovo_nb = OneVsOneClassifier(gs_nb)
ovo_nb.fit(X_train_scaled, y_train)
y_pred = ovo_nb.predict(X_test_scaled)
ovo_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovo_score)

training_results.append({"model": "naive-bayes", "strategy": "one-vs-one", "time": training_time, "accuracy": ovo_score})


# In[40]:


# OutputCodeClassifier

start = time.time()
oc_nb = OutputCodeClassifier(gs_nb)
oc_nb.fit(X_train_scaled, y_train)
y_pred = oc_nb.predict(X_test_scaled)
oc_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", oc_score)

training_results.append({"model": "naive-bayes", "strategy": "output-code", "time": training_time, "accuracy": oc_score})


# #### Decision tree

# In[41]:


from sklearn.tree import DecisionTreeClassifier

parameters = {"max_leaf_nodes": list(range(2, 100)), "min_samples_split": [2, 3, 4]}
gs_dt = GridSearchCV(DecisionTreeClassifier(random_state=123), parameters, scoring="roc_auc", cv=10)


# In[42]:


# OneVsRestClassifier

start = time.time()
ovr_dt = OneVsRestClassifier(gs_dt)
ovr_dt.fit(X_train_scaled, y_train)
y_pred = ovr_dt.predict(X_test_scaled)
ovr_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovr_score)

training_results.append({"model": "decision-tree", "strategy": "one-vs-rest", "time": training_time, "accuracy": ovr_score})


# In[43]:


# OneVsOneClassifier

start = time.time()
ovo_dt = OneVsOneClassifier(gs_dt)
ovo_dt.fit(X_train_scaled, y_train)
y_pred = ovo_dt.predict(X_test_scaled)
ovo_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", ovo_score)

training_results.append({"model": "decision-tree", "strategy": "one-vs-one", "time": training_time, "accuracy": ovo_score})


# In[44]:


# OutputCodeClassifier

start = time.time()
oc_dt = OutputCodeClassifier(gs_dt)
oc_dt.fit(X_train_scaled, y_train)
y_pred = oc_dt.predict(X_test_scaled)
oc_score = accuracy_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", oc_score)

training_results.append({"model": "decision-tree", "strategy": "output-code", "time": training_time, "accuracy": oc_score})


# - [x] Compare training time and quality of all models and all strategies. To conclude.

# ### Comparing models

# In[45]:


tr = pd.DataFrame.from_records(training_results)
tr = tr.set_index(["model", "strategy"])
tr


# In[46]:


tr_time = []
tr_accuracy = []

for m in ["logistic-regression", "svm", "knn", "naive-bayes", "decision-tree"]:
    tmp_time = {"model": m}
    tmp_accuracy = {"model": m}
    for s in ["one-vs-rest", "one-vs-one", "output-code"]:
        tmp_time[s] = tr.loc[m, s]["time"]
        tmp_accuracy[s] = tr.loc[m, s]["accuracy"]
    tr_time.append(tmp_time)
    tr_accuracy.append(tmp_accuracy)


# In[47]:


df_time = pd.DataFrame.from_records(tr_time)
df_time = df_time.set_index(["model"])
df_time


# In[48]:


sns.heatmap(df_time, annot=True, cmap="crest")


# In[49]:


training_results[0]["model"]


# In[50]:


df_accuracy = pd.DataFrame.from_records(tr_accuracy)
df_accuracy = df_accuracy.set_index(["model"])
df_accuracy


# In[51]:


sns.heatmap(df_accuracy, annot=True, cmap="crest")


# ### The conclusion
# 
# The goal of this project was to improve [the previous results](https://github.com/Mark-Rozenberg/Crime-And-Climate) on this dataset provided by Mark Rozenberg which were 29.5% accuracy with Multi-layer perceptron model, without the climate features.
# 
# We followed these steps to accomplish this goal:
# - Resampling
# - Hyperparameter optimization
# - Reduce dimensionality (both in records and in classes)
# 
# #### Our experiments showed that the Decision tree model with one-vs-one strategy was the highest performer with 52% accuracy. Here we should consider that our class count was 3, while the original experimenter trained for 10 classes. Overall, our and Mark's experiments show that it is hard to predict crime types given the features.

# # Multilabel classification

# - [x] (**+3 points**) Repeat all points for the **Multilabel classification** task (classification with multiple target features, for example, binary ones). Try **MultiOutputClassifier** and **ClassifierChain** as strategies.

# In[52]:


df


# In[53]:


X, y = df.drop(columns=["crime_type_cat", "location_desc_cat"]), df[["crime_type_cat", "location_desc_cat"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

training_results = []


# In[54]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

parameters = {
  "n_estimators": [5, 50, 200],
  "max_depth" : [None, 10,20],
  "min_samples_split" : [2, 5, 10],
}
gs_rf = GridSearchCV(RandomForestClassifier(random_state=123), parameters, n_jobs=2, cv=10)


# In[55]:


# MultiOutputClassifier

start = time.time()
mo_rf = MultiOutputClassifier(gs_rf)
mo_rf.fit(X_train_scaled, y_train)
accuracy = mo_rf.score(X_test_scaled, y_test)
training_time = time.time() - start

print("training time:", training_time)
print("accuracy:", accuracy)

training_results.append({"model": "Random Forest", "strategy": "multi-output-classifier", "time": training_time, "accuracy": accuracy})


# In[56]:


tr = pd.DataFrame.from_records(training_results)
tr = tr.set_index(["model", "strategy"])
tr


# ### The conclusion
# 
# As with Multioutput regression, the accuracy is bad. It could mean that the chosen features do not explain the target features (crime_type_cat, location_desc_cat).

# # Multioutput regression

# - [x] (**+2 points**) Repeat all steps for the **Multioutput Regression** task (multioutput regression, real). Model to try at least one: **Ridge**. Try **MultiOutputRegressor** and **RegressorChain** as strategies. Use **R2** as the metric.

# In[57]:


df


# In[58]:


X, y = df.drop(columns=["crime_type_cat", "location_desc_cat"]), df[["crime_type_cat", "location_desc_cat"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

training_results = []


# #### Ridge

# In[59]:


from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

parameters = {"alpha":[1, 10]}
gs_ridge = GridSearchCV(Ridge(random_state=123), parameters, scoring="r2", cv=10)


# In[60]:


# MultiOutputRegressor

start = time.time()
mo_regr = MultiOutputRegressor(gs_ridge)
mo_regr.fit(X_train_scaled, y_train)
y_pred = mo_regr.predict(X_test_scaled)
score = r2_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("R2 score:", score)

training_results.append({"model": "ridge", "strategy": "multi-output-regressor", "time": training_time, "accuracy": score})


# In[61]:


# RegressorChain

start = time.time()
regr_chain = RegressorChain(gs_ridge)
regr_chain.fit(X_train_scaled, y_train)
y_pred = regr_chain.predict(X_test_scaled)
score = r2_score(y_test, y_pred)
training_time = time.time() - start

print("training time:", training_time)
print("R2 score:", score)

training_results.append({"model": "ridge", "strategy": "regressor-chain", "time": training_time, "accuracy": score})


# In[62]:


tr = pd.DataFrame.from_records(training_results)
tr = tr.set_index(["model", "strategy"])
tr


# ### The conclusion
# 
# The R2 score is exceptionally low, almost 0. It could mean that the chosen features do not explain the target features (crime_type_cat, location_desc_cat).
