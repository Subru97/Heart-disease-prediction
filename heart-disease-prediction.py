import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('framingham.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

missing_value_column_names = data.columns[data.isnull().sum()>0]                  
missing_value_column_indexes = [data.columns.get_loc(c) for c in missing_value_column_names]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
imputer = imputer.fit(X[:, missing_value_column_indexes])
X[:, missing_value_column_indexes] = imputer.transform(X[:, missing_value_column_indexes])

from sklearn.feature_selection import SelectKBest, chi2
test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(X, y)
independent_features = [fit.scores_.tolist().index(i) for i in fit.scores_ if i>150 ]
X = X[:, independent_features]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression()

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier()

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10,criterion="entropy" )

# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion='entropy')

# from sklearn.svm import SVC
# classifier = SVC(kernel='linear')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.f1_score(y_test, y_pred), metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred))
