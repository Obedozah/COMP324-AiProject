import preprocess_data as pre
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

def train_and_save():
    data, scaler, columns = pre.preprocess_data('datasets/KDDTrain+_20Percent.arff')

    X = data[:, :-1]
    Y = data[:, -1]

    clf = RandomForestClassifier()
    clf = clf.fit(X, Y)

    joblib.dump(clf, 'models/random_forest_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(columns, 'models/columns.joblib')
    print("Trained + Saved")
    
    fn=data.dtype.names
    cn=data.dtype.names
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(clf.estimators_[0],
                feature_names = fn, 
                class_names=cn,
                filled = True)
    fig.savefig('rf_individualtree.png')

if __name__ == "__main__":
    train_and_save()