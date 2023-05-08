import pandas as pd
import numpy as np

# Loading Dataset
df = pd.read_csv('dataset/iris.csv', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])

# Exploring Dataset
# print(df.head())


# Selecting Features and Label
X = df[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
y = df['class']

# Splitting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('Training Cases: %d \nTest Cases: %d \n' % (X_train.shape[0], X_test.shape[0]))

# Feature Scaling - Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit Model on Training Dataset
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting Test Dataset
y_pred = model.predict(X_test)

# Evaluating Model Metrics
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred), '\n')
#print('MSE Score:', mean_squared_error(y_test, y_pred), '\n')
print('Accuracy:', accuracy_score(y_test, y_pred), '\n')
#print('Precision:', precision_score(y_test, y_pred), '\n')
#print('Recall:', recall_score(y_test, y_pred), '\n')
print('Model Score:', model.score(X_test, y_test))

# Generating Pickle File from Model
import pickle
pickle.dump(model, open('irismodel.pkl', 'wb'))


# Debugging
print('\n>>>END<<')