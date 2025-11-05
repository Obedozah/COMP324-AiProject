import preprocess_data as pre
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_and_save():
    data = pre.preprocess_data('datasets/KDDTrain+_20Percent.arff')

    X = data[:, :-1]
    Y = data[:, -1]

    clf = RandomForestClassifier()
    clf = clf.fit(X, Y)
    joblib.dump(clf, 'models/random_forest_model.joblib')
    print("Trained + Saved")

if __name__ == "__main__":
    train_and_save()