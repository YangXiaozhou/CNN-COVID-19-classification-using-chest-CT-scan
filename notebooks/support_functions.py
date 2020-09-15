import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def read_ct_scan(folder_name):
    """Read the CT scan image files from directory"""
    images = []
    
    # Construct path for two image file folders
    filepaths = [os.path.join(folder_name,file) for file in os.listdir(folder_name) if file != '.DS_Store']
    
    # Read image file in each folder and convert them to RGB channel
    for file in filepaths:
        images.append(np.array(Image.open(file).convert('RGB')))
        
    return images

def get_metric_history(history):
    """Extract loss and metrics history from model's training history,
    and return as a list that has training and validation:
    1. loss
    2. accuracy
    3. type I error
    4. type II error
    """
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


def plot_metric_history(metric_history):
    """Plot loss and metrics history"""
    epochs_range = range(len(metric_history[0]))

    plt.figure(figsize=(12, 10))
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
    plt.plot(epochs_range, metric_history[4], label='Training Type I Error')
    plt.plot(epochs_range, metric_history[5], '-', label='Validation Type I Error')
    plt.legend(loc='upper right')
    plt.title('Type I Error')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, metric_history[6], label='Training Type II Error')
    plt.plot(epochs_range, metric_history[7], '-', label='Validation Type II Error')
    plt.legend(loc='upper right')
    plt.title('Type II Error')
    plt.show()
    return 


def resize_and_shuffle(X, y, img_size=(256,256), batch_size=32, seed=123, buffer_size=500):
    """Resize image files to equal height and width of IMG_SIZE, 
    Convert data to tensorflow.data.Dataset objects with random shuffle and batches
    """
    results = list(map(lambda img: tf.image.resize(img, img_size), X))
    results = tf.data.Dataset.from_tensor_slices((results, y))
    ds = results.shuffle(len(y)).batch(batch_size)
    
    return ds
