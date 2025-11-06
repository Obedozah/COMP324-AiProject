import preprocess_data as pre

data, scaler, columns = pre.preprocess_data('datasets/KDDTest+.arff')

print(data[:5, :])