import preprocess_data as pre
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

test_data = pre.preprocess_data('datasets/KDDTest+.arff')
train_data = joblib.load('models/random_forest_model.joblib')

X_test = test_data[:, :-1]
Y_test = test_data[:, -1]

predictions = train_data.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, predictions))
print("Classification Report:\n", classification_report(Y_test, predictions))
