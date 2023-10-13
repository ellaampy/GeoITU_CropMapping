import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
            
def load_data(country, feature, mode, path):
    """
    load a numpy array for a region
    
    parameters
        country(str): name of the region
        feature(str): type of features to load
        path(str): directory to locate array
        mode(str): include
    """
    
    X = np.load(os.path.join(path, '{}_{}_{}.npy'.format(country, mode, feature)), allow_pickle=True)
    
    if mode == 'Train':
        y = np.load(os.path.join(path, '{}_{}_labels.npy'.format(country, mode)), allow_pickle=True)
        assert X.shape[0] == y.shape[0]
        return X, y
    else:
        return X

    
    
def interpolate_none_values(array):
    """
    interpolate array at None points
    
    """
    T, C = array.shape

    # iterate through each channel
    for c in range(C):
        channel = array[:, c]

        # find indices of None values
        non_none_indices = np.where(channel != None)[0]
        none_indices = np.where(channel == None)[0]

        if len(non_none_indices) > 1:
            # interpolate linearly
            f = interp1d(non_none_indices, channel[non_none_indices], kind='linear', fill_value='extrapolate')
            channel[none_indices] = f(none_indices)

    return array




def create_ml_data(npy_path, csv_path=None):
    """
    creates a 3D array by stacking 2D arrays
    
    parameters
        npy_path: directory containing several .npy files
    
    
    returns 
        a stacked array of size samples x time x channel
        labels of size, samples
        ids of size, samples
    """
    
    ## collector for arrays
    npy_collector = []
    label_collector = []
    id_collector = []
    
    # get list of .npy files in path
    npy_list = os.listdir(npy_path)
    
    # used only when label data is available e.g Train
    if csv_path is not None:
        label_df = pd.read_csv(csv_path)
    
    for i in npy_list:
        data = np.load(os.path.join(npy_path, i), allow_pickle=True)        
        
        
        try:
            # checker for nan
            data = data/100.00
        except:
            # interpolate if nan found
            data = interpolate_none_values(data)
        
        npy_collector.append(data)
        id_collector.append(i.split('.')[0])
            

        # filter label in csv. only valid for Train
        if csv_path is not None:
            # filter label in csv
            y = label_df.loc[(label_df['ID'] == i.split('.')[0])]['Target'].values[0]
            label_collector.append(y)
    
    if csv_path is not None:
        return np.stack(npy_collector), np.array(label_collector), np.array(id_collector)
    else:
        return np.stack(npy_collector), np.array(id_collector) 


    
    
def compute_indices(X_3d):
    
    """
    computes spectral indices given a 3d array 

    
    parameters
        X_3d(array): a 3D array of shape NxTxC
        N: sample size
        T: time steps
        C: number of channels
           channel order==> blue, green, red, 
                            red edge1, red edge2, red edge3,
                            near infrared, red edge4, swir1, swir2
                          
    """
    
    epsilon = 1e-10

    NDVI = np.divide(X_3d[:,:,6] - X_3d[:,:,2],  (X_3d[:,:,6] + X_3d[:,:,2]) + epsilon) # normalized difference vegetation index
    NDVI_RE = np.divide(X_3d[:,:,6] - X_3d[:,:,4],  (X_3d[:,:,6] + X_3d[:,:,4]) + epsilon) # red edge ndvi
    NDWI = np.divide(X_3d[:,:,1] - X_3d[:,:,6],  (X_3d[:,:,1] + X_3d[:,:,6]) + epsilon) # normalized difference water index
    EVI = np.divide(X_3d[:,:,6] - X_3d[:,:,2],  (X_3d[:,:,6] + X_3d[:,:,2] - (7.5 * X_3d[:,:,0]) + 1 + epsilon)) * 2.5 # enhanced vegetation index
    NDBI = np.divide(X_3d[:,:,8] - X_3d[:,:,6],  (X_3d[:,:,8] + X_3d[:,:,6]) + epsilon) # built up area index
    SAVI  = np.divide(X_3d[:,:,6] - X_3d[:,:,2],  (X_3d[:,:,6] + X_3d[:,:,2] + 0.5) + epsilon) * (1+0.5) # soil adjusted vegetation index
    BRI = np.divide(np.divide(1, X_3d[:,:,1]+ epsilon) - np.divide(1, X_3d[:,:,3]+ epsilon), X_3d[:,:,6] + epsilon) # browning reflectance index
    CI = np.divide(X_3d[:,:,6], X_3d[:,:,3]+epsilon) - 1 ## chrolophy index
    LSWI = np.divide(X_3d[:,:,6] - X_3d[:,:,8], (X_3d[:,:,6] + X_3d[:,:,8]) + epsilon)# land surface water index
    NDPI = np.divide(X_3d[:,:,5] - X_3d[:,:,1], (X_3d[:,:,5] + X_3d[:,:,1]) + epsilon) # normalized difference pond index
    WRI =  np.divide(X_3d[:,:,1] + X_3d[:,:,2], (X_3d[:,:,6] + X_3d[:,:,9]) + epsilon) # water ration index
    PSRI = np.divide(X_3d[:,:,2] + X_3d[:,:,2], X_3d[:,:,6]+  epsilon) # plant senescence reflectance index
    DVI  = X_3d[:,:,6] - X_3d[:,:,2] # difference vegetation index
    RVI = np.divide(X_3d[:,:,2] , X_3d[:,:,6]+ epsilon) # ratio vegetation index
    VARI = np.divide(X_3d[:,:,1] - X_3d[:,:,2],  (X_3d[:,:,1] + X_3d[:,:,2] - X_3d[:,:,0] )+epsilon)

    list_features = [NDVI, NDVI_RE, NDWI, EVI, NDBI, SAVI, BRI, CI, LSWI, NDPI, WRI, PSRI, DVI, RVI, VARI]
    X_indices = np.moveaxis(np.stack(list_features), 0, -1)
    
    return X_indices