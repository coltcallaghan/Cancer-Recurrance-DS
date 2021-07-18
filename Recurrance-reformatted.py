# This uses the original .xls file

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import your dependencies 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from io import StringIO

# Importing the dataset
dataset = pd.read_excel('breast-cancer.xls')
dataset1 = dataset.drop("tumor-size", axis=1)
dataset1 = dataset1.drop("inv-nodes", axis=1)
X = dataset1.iloc[:, :-1].values
y = dataset1.iloc[:, -1].values


# Encoding y
le = LabelEncoder()
y = le.fit_transform(y)

# Encoding X
ct1 = ColumnTransformer(transformers=[('age', OneHotEncoder(), [0]),('deg-malig', OneHotEncoder(), [3]),('breast-quad', OneHotEncoder(), [5])], remainder='passthrough')
X = np.array(ct1.fit_transform(X))
ct2 = ColumnTransformer(transformers=[('menopause', OneHotEncoder(), [15]),('node-caps', OneHotEncoder(), [16]),('breast', OneHotEncoder(), [17]),('irridiat', OneHotEncoder(), [18])], remainder='passthrough')
X = np.array(ct2.fit_transform(X))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# Applying StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
classifier = SVC(kernel = "linear")
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test) 

# Making a Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Unseen data     
input_text = """40-49,ge40,yes,1,left,left_low,no"""
inp = pd.read_csv(StringIO(input_text), names = ['age', 'menopause', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])
inp = np.array(ct1.transform(inp))
inp = np.array(ct2.transform(inp))
inp = sc.transform(inp)

prediction = classifier.predict(inp)

# Finding the accuracy
def accuracy(cm):
    diagonal = cm.trace()
    sum_all = cm.sum()
    return diagonal / sum_all

accuracy = accuracy(cm)
