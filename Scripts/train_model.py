import preprocess_data as pre
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

def train_and_save():
    data, scaler, columns = pre.preprocess_data('datasets/KDDTrain+.arff')

    X = data[:, :-1]
    Y = data[:, -1]

    #clf = RandomForestClassifier()
    #clf = clf.fit(X, Y)

    #joblib.dump(clf, 'models/random_forest_model.joblib')
    #joblib.dump(scaler, 'models/scaler.joblib')
    #joblib.dump(columns, 'models/columns.joblib')

    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )

    clf.fit(X, Y)
    joblib.dump(clf, 'models/svm_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(columns, 'models/columns.joblib')
    print("SVM trained + saved")


if __name__ == "__main__":
    train_and_save()