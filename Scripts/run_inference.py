import preprocess_data as pre
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

clf = joblib.load('models/svm_model.joblib')
scaler = joblib.load('models/scaler.joblib')
columns = joblib.load('models/columns.joblib')

test_data, _, _ = pre.preprocess_data('datasets/KDDTest+.arff', fit_scaler = scaler, fit_columns = columns)

X_test = test_data[:, :-1]
Y_test = test_data[:, -1]

predictions = clf.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, predictions))
print("Classification Report:\n", classification_report(Y_test, predictions, digits=10))