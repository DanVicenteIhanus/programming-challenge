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
    
    # find PCA dimension using the MLE
    pca = PCA(n_components = 'mle', svd_solver='auto')
    pca_data = pca.fit_transform(scaled_data)
    
    return pca_data

def remove_outliers(data: pd.DataFrame, ZSCORE_THREASHOLD: int = 4) -> pd.DataFrame:
    zscore = np.abs(stats.zscore(data.select_dtypes(include=["float", "int"])))
    is_inlier = ~ (zscore > ZSCORE_THREASHOLD).any(axis=1)
    data = data[is_inlier]
    return data


'''
------------------------------------
TODO: Pick some other model (NN?)
------------------------------------
'''

def ann_routine(features, labels, K1, K2, a1, a2, k1_init, k2_init, opt, Nbatch, epochs):
    one_hot_labels = to_categorical(labels, num_classes=3)
    input_layer = tf.keras.Input(shape=features.shape[1:])
    Z1 = keras.layers.Dense(units=K1, activation=a1, use_bias=True,
                            kernel_initializer=k1_init,
                            bias_initializer='zeros',
                            name='hidden_layer_1')
    Z2 = keras.layers.Dense(units=K2, activation=a2, use_bias=True,
                            kernel_initializer=k2_init,
                            bias_initializer='zeros',
                            name='hidden_layer_2')
    
    output_layer = keras.layers.Dense(units=3, 
                                      activation='softmax',
                                        name='output_layer')

    model = keras.models.Sequential([input_layer, Z1, Z2, output_layer])
    
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    pred = model.fit(x=features, y=one_hot_labels, batch_size=Nbatch,
                     epochs=epochs,
                     validation_split=0.3,
                     verbose=1)
    return pred

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

    # --- Visualize classes --- #
    print('================ Class distribution ==================')
    classes = labels.unique()
    print(f'Available classes: {classes}')
    for cl in classes:
        print(f'Amount in class {cl}: {labels.value_counts()[cl]}')
    print('======================================================')

    # --- Hyperparameters --- #
    K1 = 10 # Nodes in layer 1
    K2 = 10 # Nodes in layer 2
    a1 = 'sigmoid'
    a2 = 'relu'
    k1_init = 'random_uniform'
    k2_init = 'random_normal'
    epochs = 100 # EPOCHS
    NBatch = 50
    learning_rate = 0.01
    opt = optimizers.Adamax(learning_rate=learning_rate)
    
    blahonga = ann_routine(pca_features, labels, K1=K1,
                 K2=K2, opt=opt, a1=a1, a2=a2, 
                 k1_init=k1_init, k2_init=k2_init,
                 Nbatch=NBatch, epochs=epochs)
    
