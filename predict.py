import numpy as np
import csv
import sys

from validate import validate
import concurrent.futures

train_X_file_path = "./train_X_pr.csv"
train_Y_file_path = "./train_Y_pr.csv"
validation_split = 0.2
numClasses = 2
tolerance = 0.00000001
maxEpochs = 1000000
learningRate = 0.085
categorical_column_indices = [0, 3]
numerical_column_indices = [1, 2, 4, 5, 6]
Lambda = 0.25

def remove_rows_with_null_values(X):
    return X[~np.isnan(X).any(axis=1),:]

def replace_null_values_with_zeros(X):
    X[np.isnan(X)] = 0
    return X

def replace_null_values_with_mean(X):
    colMeans = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    
    X[inds] = colMeans[inds[1]]
    return X

def standardize(X, column_indices):
    colMeans = np.nanmean(X, axis=0)
    colSTDs = np.nanstd(X, axis=0)
    for col in column_indices:
        X[:,col] = (X[:,col] - colMeans[col])/colSTDs[col]
    return X

def min_max_normalize(X, column_indices):
    colMins = np.min(X, axis=0)
    colMaxs = np.max(X, axis=0)
    for col in column_indices:
        X[:,col] = (X[:,col]-colMins[col])/(colMaxs[col] - colMins[col])
    return X

def mean_normalize(X, column_indices, colMeans=[], colMins=[], colMaxs=[]):
    if len(colMeans)==0 or len(colMaxs)==0 or len(colMins)==0:
        cols = X[:, column_indices]
        colMeans = np.mean(cols, axis=0)
        colMaxs = np.max(cols, axis=0)
        colMins = np.min(cols, axis=0)
        X[:,column_indices] = (cols-colMeans)/(colMaxs-colMins)
        return X, colMeans, colMins, colMaxs
    else:
        cols = X[:, column_indices]
        X[:,column_indices] = (cols-colMeans)/(colMaxs-colMins)
        return X

def convert_to_numerical_labels(X):
    els = np.unique(X)
    newX = []
    for e in X:
        newX.append(np.where(els == e)[0][0])
    return newX

def apply_one_hot_encoding(X):
    els = np.unique(X)
    newX = []
    for r in X:
        newR = np.zeros(els.shape[0], dtype=np.int64)
        newR[np.where(els == r)[0][0]] = 1 
        newX.append(newR)
    return newX

def convert_given_cols_to_one_hot(X, column_indices):
    newX = np.copy(X)
    colOffset = 0
    for col in column_indices:
        enc = (apply_one_hot_encoding(X[:,col+colOffset]))
        if col == 0:
            X = np.concatenate((enc, X[:,(col+1):]),axis=1)
        elif col == len(X[0])-1:
            X = np.concatenate((X[:,0:(col+colOffset)], enc), axis=1)
        else:
            newX = np.concatenate((X[:,0:(col+colOffset)], enc), axis=1)
            X = np.concatenate((newX, X[:,(col+1+colOffset):]),axis=1)
        colOffset += (len(enc[0])-1)
    return X

def get_correlation_matrix(X, Y):
    newX = np.zeros((X.shape[0], X.shape[1]+1))
    newX[:, 0] = Y[:,0]
    newX[:,1:] = X  
    return np.corrcoef(newX.T)

def select_features(corr_mat, T1, T2):
    toSel, toRem = [], []
    for i in range(1,len(corr_mat[:,0])):
        if abs(corr_mat[i][0]) > T1:
            toSel.append(i-1)
    for i in range(len(toSel)):
        for j in range(i+1, len(toSel)):
            f1 = toSel[i]
            f2 = toSel[j]
            if f1 not in toRem and f2 not in toRem:
                if abs(corr_mat[f1][f2]) > T2:
                    toRem.append(f2)
    for r in toRem:
        toSel.remove(r)               
    return toSel

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def computeCost(X, Y, W, b, Lambda):
    A = sigmoid(np.dot(X,W)+b)
    A[A == 1] = 0.999
    A[A == 0] = 0.001
    return (-np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) + (Lambda/2)*np.sum(W**2))/(X.shape[0])

def getGrads(X, Y, W, b, Lambda):
    A = sigmoid(np.dot(X,W)+b)
    A[A == 1] = 0.999
    A[A == 0] = 0.001
    dW = (np.dot(X.T,(A-Y)) + (Lambda/2)*W)/X.shape[0]
    dB = np.sum(A-Y)
    return (dW, dB)

def getF1score(X, Y, W, b):
    Yhat = sigmoid(np.dot(X,W)+b)
    Yhat[Yhat == 1] = 0.999
    Yhat[Yhat == 0] = 0.001
    Yhat = (Yhat >= 0.5).astype(int)
    from sklearn.metrics import f1_score
    return f1_score(Y, Yhat, average = 'weighted')


"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def predict_target_values(test_X, weights):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.

    """
    Note:
    The preprocessing techniques which are used on the train data, should also be applied on the test 
    1. The feature scaling technique used on the training data should be applied as it is (with same mean/standard deviation/min/max) on the test data as well.
    2. The one-hot encoding mapping applied on the train data should also be applied on test data during prediction.
    3. During training, you have to write any such values (mentioned in above points) to a file, so that they can be used for prediction.
     
    You can load the weights/parameters and the above mentioned preprocessing parameters, by reading them from a csv file which is present in the preprocessing_regularization.zip
    """
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in preprocessing_regularization.zip and imported properly.
    """
    W, b = weights[1:] , weights[0] 
    out = [0 if hx < 0 else 1 for  hx in (np.dot(test_X,W)+b)]
    return np.array(out)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def trainValSplit(X,Y):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    valIndex = -int(validation_split*(train_X.shape[0]))
    val_X = train_X[valIndex:]
    val_Y = train_Y[valIndex:]
    train_X = train_X[:valIndex]
    train_Y = train_Y[:valIndex]
    return (train_X, train_Y, val_X, val_Y)

def buildModel(X, Y):
    train_X, train_Y, val_X, val_Y = trainValSplit(X,Y)
    W = np.random.normal(0, 1, (X.shape[1], 1))
    b = np.random.rand() 
    numEpochs = 0
    prevCost = 0
    while numEpochs < maxEpochs:
        dW, dB = getGrads(train_X, train_Y, W, b, Lambda) 
        W = W - learningRate*(dW) 
        b = b - learningRate*(dB)  
        curCost = computeCost(val_X, val_Y, W, b, Lambda)
        if numEpochs%1000 == 0:
            print(f"{numEpochs} => {curCost}")
        if abs(curCost - prevCost) < tolerance:
            break
        if curCost < 0.7:
            numEpochs += 1
        prevCost = curCost
    return (W, b)

def preprocessTrainData(X, Y):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    train_X = replace_null_values_with_mean(train_X)
    corrMat = get_correlation_matrix(train_X, train_Y)
    T1, T2 = np.mean(corrMat[:,0])/2, 0.4
    selFeatures = select_features(corrMat,T1, T2)
    for c in categorical_column_indices:
        if c not in selFeatures:
            categorical_column_indices.remove(c)
    for c in numerical_column_indices:
        if c not in selFeatures:
            numerical_column_indices.remove(c)
    train_X, train_X_means, train_X_mins, train_X_maxs = mean_normalize(train_X, numerical_column_indices)
    train_X = train_X[:, selFeatures]
    train_X = convert_given_cols_to_one_hot(train_X, categorical_column_indices)
    return (train_X, train_Y, train_X_means, train_X_mins, train_X_maxs, selFeatures)

def preprocessTestData(X, train_X_means, train_X_mins, train_X_maxs, selFeatures):
    train_X = np.copy(X)
    train_X = replace_null_values_with_mean(train_X)
    train_X = mean_normalize(train_X, numerical_column_indices, train_X_means, train_X_mins, train_X_maxs)
    train_X = train_X[:, selFeatures]
    train_X = convert_given_cols_to_one_hot(train_X, categorical_column_indices)
    return (train_X)

def predict(test_X_file_path):
    train_X = np.genfromtxt(train_X_file_path, delimiter=",", dtype=np.float128, skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter="\n", dtype=np.int64, skip_header=0)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y).reshape(train_X.shape[0],1)
    train_X, train_Y, train_X_means, train_X_mins, train_X_maxs, selFeatures = preprocessTrainData(train_X, train_Y)

    X = np.copy(train_X)
    Y = np.copy(train_Y)
    W, b = buildModel(X, Y)
    theta = [b]
    theta.extend(W)

    theta = np.array(theta, dtype=object)
    np.savetxt("WEIGHTS_FILE.csv",theta,delimiter=",")
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    test_X = preprocessTestData(test_X, train_X_means, train_X_mins, train_X_maxs, selFeatures)
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")

if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 