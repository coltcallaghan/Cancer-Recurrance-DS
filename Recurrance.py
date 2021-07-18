# This uses an amended .xls files where the dates were removed to leave NaN. 
# This allowed me to use the mode for tumor-size and inv-nodes so I didn't lose a large amount of data.
# Doing this provided better accuracy however I do not know how you would like to recieve the .xls so 
# I also added a reformatted version where I used pandas to remove the problem columns 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import your dependencies
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from io import StringIO

# Importing the dataset
dataset = pd.read_excel('breast-cancer-2.xls')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Change missing data to the mode 

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, 2:4])
X[:, 2:4] = imputer.transform(X[:, 2:4])

# Encoding y
le = LabelEncoder()
y = le.fit_transform(y)

# Encoding X
ct1 = ColumnTransformer(transformers=[('age', OneHotEncoder(), [0]),('tumor-size', OneHotEncoder(), [2]),('deg-malig', OneHotEncoder(), [5]),('breast-quad', OneHotEncoder(), [7])], remainder='passthrough')
X = np.array(ct1.fit_transform(X))
ct2 = ColumnTransformer(transformers=[('menopause', OneHotEncoder(), [24]),('inv-nodes', OneHotEncoder(), [25]),('node-caps', OneHotEncoder(), [26]),('breast', OneHotEncoder(), [27]),('irridiat', OneHotEncoder(), [28])], remainder='passthrough')
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
input_text = """40-49,ge40,15-19,0-2,yes,1,left,left_low,no"""
inp = pd.read_csv(StringIO(input_text), names = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])
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
