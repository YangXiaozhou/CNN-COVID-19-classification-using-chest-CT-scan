# Functions for Model.py use
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics

import cv2
np.seterr(divide='ignore', invalid='ignore')
from skimage.feature import blob_dog,greycomatrix, greycoprops, local_binary_pattern
from sklearn.cluster import KMeans
from math import sqrt


def read_ct_scan(folder_name):
    """Read the CT scan image files from directory"""
    images = []
    
    # Construct path for two image file folders
    filepaths = [os.path.join(folder_name,file) for file in os.listdir(folder_name) if file != '.DS_Store']
    
    # Read image file in each folder and convert them to RGB channel
    for file in filepaths:
        images.append(np.array(Image.open(file).convert('RGB')))
        
    return images

def read_github_ct_scan(folder_name,figure_name):
    """Read the CT scan image files from directory"""
    images = []

    # Construct path for two image file folders
    filepaths = [os.path.join(folder_name,file) for file in figure_name if file != '.DS_Store']

    # Read image file in each folder and convert them to RGB channel
    for file in filepaths:
        images.append(np.array(Image.open(file).convert('RGB')))

    return images

def get_metric_history(history):
    keys = [key for key in history.history.keys()]    
    
    loss=history.history[keys[0]]
    val_loss=history.history[keys[6]]
    
    acc = history.history[keys[1]]
    val_acc = history.history[keys[7]]

    false_pos = np.array(history.history[keys[2]])
    true_pos = np.array(history.history[keys[3]])
    false_negs = np.array(history.history[keys[4]])
    true_negs = np.array(history.history[keys[5]])

    val_false_pos = np.array(history.history[keys[8]])
    val_true_pos = np.array(history.history[keys[9]])
    val_false_negs = np.array(history.history[keys[10]])
    val_true_negs = np.array(history.history[keys[11]])

    type_1 = np.nan_to_num(false_pos/(false_pos+true_negs), nan=0)
    val_type_1 = np.nan_to_num(val_false_pos/(val_false_pos+val_true_negs))

    type_2 = np.nan_to_num(false_negs/(false_negs+true_pos))
    val_type_2 = np.nan_to_num(val_false_negs/(val_false_negs+val_true_pos))
    
    metric_history = [loss, val_loss, acc, val_acc, type_1.tolist(), 
                      val_type_1.tolist(), type_2.tolist(), val_type_2.tolist()]
    return metric_history


def plot_metric_history(metric_history, epochs):
    """Plot metrics during model training"""
    epochs_range = range(epochs)

    plt.figure(figsize=(15, 12))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, metric_history[0], label='Training Loss')
    plt.plot(epochs_range, metric_history[1], '-', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, metric_history[2], label='Training Accuracy')
    plt.plot(epochs_range, metric_history[3], '-', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, metric_history[4], label='Training Type-1 Error')
    plt.plot(epochs_range, metric_history[5], '-', label='Validation Type-1 Error')
    plt.legend(loc='upper right')
    plt.title('Type-1 Error')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, metric_history[6], label='Training Type-2 Error')
    plt.plot(epochs_range, metric_history[7], '-', label='Validation Type-2 Error')
    plt.legend(loc='upper right')
    plt.title('Type-2 Error')
    plt.show()
    return 

    
# ----------------------------------------------------------------------------------
# Functions for submission
def resize_and_shuffle(X, y, img_size=(256,256), batch_size=32, seed=123, buffer_size=500):
    """Resize image files to equal height and width of IMG_SIZE, 
    Convert data to tensorflow.data.Dataset objects with random shuffle and batches
    """
    results = list(map(lambda img: tf.image.resize(img, img_size), X))
    results = tf.data.Dataset.from_tensor_slices((results, y))
    ds = results.shuffle(len(y)).batch(batch_size)
    
    return ds

def convert_class_label(y):
    y = np.array(y)
    y[y == 'COVID'] = 1
    y[y != '1'] = 0
    y = y.astype(int)
    return y

def create_data_augment_layer(random_seed):
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=random_seed),
            layers.experimental.preprocessing.RandomRotation((-0.05, 0.05), seed=random_seed),
            layers.experimental.preprocessing.RandomZoom(-0.05, seed=random_seed),
            layers.experimental.preprocessing.RandomTranslation(0.05, 0.05, seed=random_seed),
            layers.experimental.preprocessing.RandomContrast(0.05, seed=random_seed),
        ]
    )
    return data_augmentation

def label_predictions(predictions):
    predictions = predictions.reshape(len(predictions))
    mask = predictions > 0.5
    predictions = np.full(len(predictions), 'NonCOVID')
    predictions[mask] = 'COVID'
    
    return(predictions.tolist())

def feature_clustering_train(images,n_clusters, detector=cv2.AKAZE_create(),img_size=(256,256)):

    """Read the CT scan image files from directory"""
    description1 = []; description2 = []   
    for img in images:
        img = cv2.resize(img, img_size)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)        
        kp, descriptor = detector.detectAndCompute(gray, None)
        description1.extend([list(kp[i].pt)+list(descriptor[i]) for i in range(len(kp))])
        descriptor2 = blob_dog(gray, max_sigma=30, threshold=0.1,exclude_border=50)
        descriptor2[:, 2] = descriptor2[:, 2] * sqrt(2)
        description2.extend(list(descriptor2))
    Kmean = KMeans(n_clusters=n_clusters); Kmean.fit(description1)
    Kmean2 = KMeans(n_clusters=n_clusters); Kmean2.fit(description2)
    return Kmean, Kmean2

def bag_of_words(images, model1,model2,detector=cv2.AKAZE_create(),img_size=(256,256)):
    words = []
    
    for img in images:
        img = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)
        matrix_coocurrence = greycomatrix(gray, [1,2,3], [0,np.pi/4,np.pi/2,3*np.pi/4], normed=False, symmetric=False)
        glcm = np.concatenate((greycoprops(matrix_coocurrence, prop = 'contrast').ravel(),
                greycoprops(matrix_coocurrence, 'dissimilarity').ravel(),
                greycoprops(matrix_coocurrence, 'homogeneity').ravel(),
                greycoprops(matrix_coocurrence, 'ASM').ravel(),
                greycoprops(matrix_coocurrence, 'energy').ravel(),
                greycoprops(matrix_coocurrence, 'correlation').ravel()
                ))
        lbp = local_binary_pattern(gray,P=8,R=1)
        feature = np.concatenate((glcm,np.histogram(lbp.ravel(),bins=256)[0]))
        kp, descriptor1 = detector.detectAndCompute(gray, None)
        descriptor2 = blob_dog(gray, max_sigma=30, threshold=0.1,exclude_border=50)
        descriptor2[:, 2] = descriptor2[:, 2] * sqrt(2)
        label1 = model1.predict([list(kp[i].pt)+list(descriptor1[i]) for i in range(len(kp))])    
        label2 = model2.predict(descriptor2)   
        label_all = np.concatenate((np.histogram(label1,bins=model1.n_clusters)[0],
                                    np.histogram(label2,bins=model2.n_clusters)[0]))
        words.append(np.concatenate((label_all,feature)))
    return words

def prob_cnn_predict(X_test,model): 
    results = list(map(lambda img: tf.image.resize(img, (256,256)), X_test))
    resized_X = tf.data.Dataset.from_tensor_slices(results).batch(32)
    prediction = model.predict(resized_X)
    return(prediction)