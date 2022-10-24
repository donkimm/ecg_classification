# Contains functions related to data processing
import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Returns the data and labels in filepath for the mitbih_*.csv files
    """
    data = pd.read_csv(filepath, header=None)
    labels = data.pop(len(data.columns)-1) # Final column contains label data
    
    data = data.to_numpy()
    labels = labels.to_numpy()
    
    return data, labels

def get_single_input(data, labels, object_class):
    indices = np.where(labels==object_class)
    choice  = np.random.choice(indices[0], size=1)
    return data[choice[0],:,:]

def get_class_data(data, labels, object_class):
    indices = np.where(labels==object_class)
    class_data = data[indices,:,:]
    return class_data[0,:,:,0]
    