import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import tensorflow as tf
import keras
from keras import optimizers
from keras.utils import to_categorical


'''
------------------------------------
Handle data 
------------------------------------
'''

def import_data(datadir):
    return pd.read_csv(datadir, index_col=[0])

def clean_data(data: pd.DataFrame):
    data.drop_duplicates(inplace=True)
    data = data[data.y != ' arkitekt']

    features = data.drop(['y', 'x12'], axis=1)
    labels = data['y']
    le = LabelEncoder()
    encoded_col = le.fit_transform(features['x7'])
    features['x7'] = encoded_col
    features = features.fillna(0)
    labels = labels.fillna(0)
    labels.replace('Bobborg', 'Boborg',inplace=True)
    labels.replace('Jorggsuto', 'Jorgsuto', inplace=True)
    labels.replace('Atsutoob', 'Atsutobob', inplace=True)
    labels.replace('Boborg', 0, inplace=True)
    labels.replace('Jorgsuto', 1, inplace=True)
    labels.replace('Atsutobob',2, inplace=True)
    return features, labels, len(data.index)

def compute_PCA(data):
    std_scaler = StandardScaler()
    scaled_data = std_scaler.fit_transform(data)
    pca = PCA(n_components = 'mle', svd_solver='auto') # find PCA dimension using the MLE
    pca_data = pca.fit_transform(scaled_data)
    
    return pca_data

'''
------------------------------------
TODO: Pick some other model (NN?)
------------------------------------
'''

def ann_routine(features, labels, K1, K2, K3, a1, a2, a3, k1_init, k2_init, k3_init, opt, Nbatch, epochs):
    one_hot_labels = to_categorical(labels, num_classes=3)
    input_layer = tf.keras.Input(shape=features.shape[1:])
    
    Z1 = keras.layers.Dense(units=K1, activation=a1, use_bias=True,
                            kernel_initializer=k1_init,
                            bias_initializer='zeros',
                            name='hidden_layer_1')
    Z2 = keras.layers.Dense(units=K2, activation=a2, use_bias=True,
                            kernel_initializer=k2_init,
                            bias_initializer='zeros',
                            kernel_regularizer=tf.keras.regularizers.L2(0.001),
                            name='hidden_layer_2')
    Z3 = keras.layers.Dense(units=K3, activation=a3, use_bias=False,
                            kernel_initializer=k3_init,
                            name='hidden_layer_3')
    d = keras.layers.Dropout(0.2)
    output_layer = keras.layers.Dense(units=3, 
                                      activation='softmax',
                                        name='output_layer')
    model = keras.models.Sequential([input_layer, Z1,d, Z2, d, Z3, d, output_layer])
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                    metrics=['accuracy', 'categorical_crossentropy',
                             keras.metrics.Precision()])
    '''
    history = model.fit(x=features, y=one_hot_labels, batch_size=Nbatch,
                     epochs=epochs,
                     verbose=0)
    '''
    history = None
    return model, history

def svm_routine(train_features, train_labels, 
                test_features, 
                kernel, parameter, coef0):
    
    # --- kernel choice ---
    if kernel == 'linear':
        svm_model = SVC(kernel=kernel)
    if kernel == 'poly': 
        svm_model = SVC(kernel=kernel, gamma = parameter, coef0 = coef0)
    else:
        svm_model = SVC(kernel=kernel, gamma = parameter)
    
    # --- fit data and predict --- 
    svm_model.fit(train_features, train_labels)
    predicted_labels = svm_model.predict(test_features)

    return predicted_labels

def compute_accuracy(predicted, true):
    return np.mean(true == predicted)

'''
------------------------------------
Main script
------------------------------------
'''

if __name__ == '__main__':
    
    # --- Handle Data --- #
    datadir = '/Users/danvicente/Datalogi/DD2421 - Maskininlärning/programming challenge/'
    training_set = 'TrainOnMe.csv'
    test_set ='EvaluateOnMe.csv'
    training_data = import_data(datadir+training_set)
    test_data = import_data(datadir+test_set)
    features, labels, N = clean_data(training_data) # clean data
    pca_features = compute_PCA(features)            # compute PCA 
    std = StandardScaler()
    norm_features = std.fit_transform(features)

    # --- Visualize classes --- #
    print('================ Class distribution ==================')
    classes = labels.unique()
    print(f'Available classes: {classes}')
    for cl in classes:
        print(f'Amount in class {cl}: {labels.value_counts()[cl]}')
    print('======================================================')

    # --- Hyperparameters --- #
    
    # NN
    K1 = 40
    K2 = 25 
    K3 = 30
    a1 = 'relu'
    a2 = 'tanh'
    a3 = 'sigmoid'
    k1_init = 'random_uniform'
    k2_init = 'random_uniform'
    k3_init = 'random_uniform'
    epochs = 100
    NBatch = 1
    learning_rate = 0.001
    opt = optimizers.Adamax()
    
    # SVM
    kernel = 'rbf'
    gamma = 0.2

    model, history = ann_routine(pca_features, labels, K1=K1, K2=K2, K3=K3, a1=a1, a2=a2, a3=a3,
                            k1_init=k1_init, k2_init=k2_init, k3_init=k3_init,
                            opt=opt, Nbatch=NBatch, epochs=epochs)
    
    # --- 10-fold cross validation --- #
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    cv_svm = cross_val_score(SVC(kernel=kernel, gamma=gamma), X=pca_features, y=labels, cv=skf, verbose=1)
    print(f'Accuracy of SVM: {np.mean(cv_svm)}') 

    acc = []
    for train_indices, test_indices in skf.split(pca_features, labels):
        model.fit(pca_features[train_indices], to_categorical(labels[train_indices], num_classes=3),
              batch_size=NBatch, epochs=epochs, verbose=0)
        y_pred = model.predict(pca_features[test_indices])
        accuracy = compute_accuracy(np.argmax(y_pred, axis=1), labels[test_indices])
        acc.append(accuracy)

    print(f'Accuracy of ANN: {np.mean(acc)}')

    # --- Train model --- #   

    # --- Predict --- #
