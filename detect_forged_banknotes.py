# You can find the blog post about this code at: www.gabriel-berardi.com/post/detect-forged-banknotes-with-a-logistic-regression

# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Loading the dataset from https://archive.ics.uci.edu/ml/datasets/banknote+authentication

data = pd.read_csv('data_banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
print(data.head())

# Show information about all features

print(data.info())

# Use pairplot to get an overview of the features

sns.pairplot(data, hue='auth')
plt.show()

# Display a correlation heatmap of all features

mask = np.zeros(data.corr().shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
plt.figure(figsize=(7,6))
plt.title('Correlation Heatmap of All Features', size=18)
ax = sns.heatmap(data.corr(), cmap='coolwarm', vmin=-1, vmax=1,
                 center=0, mask=mask, annot=True)
plt.show()

# Show the distribution of the target

plt.figure(figsize=(8,6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=data['auth'])
target_count = data.auth.value_counts()
plt.annotate(s=target_count[0], xy=(-0.04,10+target_count[0]), size=14)
plt.annotate(s=target_count[1], xy=(0.96,10+target_count[1]), size=14)
plt.ylim(0,900)
plt.show()

# Balance the dataset with regard to the target feature

nb_to_delete = target_count[0]-target_count[1]
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
data = data[nb_to_delete:]
print(data['auth'].value_counts())

# Split our data into a training and test data set

X = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features. Note: only fit the scaler on training data to prevent data leakage

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model

clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf.fit(X_train, y_train.values.ravel())

# Make predictions on the test data

y_pred = np.array(clf.predict(X_test))

# Print a confusion matrix and calculate accuracy

conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                        columns=['Pred. Negative', 'Pred. Positive'], 
                        index=['Act. Negative', 'Act. Positive'])

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = round((tn+tp)/(tn+fp+fn+tp),4)

print(conf_mat)
print(f'\nAccuracy  = {round(100*accuracy,2)}%')

# Simulate the prediction of a single new banknote

new_banknote = np.array([4.5, -8.1, 2.4, 1.4], ndmin=2)
new_banknote = scaler.transform(new_banknote)
print(f'Prediction:         Class {clf.predict(new_banknote)[0]}')
print(f'Probability [0/1] : {clf.predict_proba(new_banknote)[0]}')
