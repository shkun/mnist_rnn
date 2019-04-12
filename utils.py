import numpy as np

def read_labels(path, filename):

    """
    inputs:
    path - full file path name
    filename - file name with extension of the binary file 
    
    returns:
    numpy array of labels (0 - 9) 
    
    """    
    
    labeldata = None
    hd_type = np.dtype('>u4') # unsigned int32 with big endien
    dt_type = np.dtype('>u1') # unsigned int8 with big endient
    
    with open(path + filename) as f:
        # Valid MNIST Label file checker
        magic = np.fromfile(f, dtype=hd_type, count=1)
        if 2049 != magic:
            print('Magic number {} incorrect. Invalid MNIST label file'.format(magic))
        else:
            # Reshape data into 1D array (rows)
            r = int(np.fromfile(f, dtype=hd_type, count=1))
            print('Reading {} with shape: ({},)'.format(filename, r))
            labeldata = np.fromfile(f, dtype=dt_type, count=r)

    return labeldata 

def read_images(path, filename):
    
    """
    inputs:
    path - full file path name
    filename - file name with extension of the binary file 
    
    returns:
    numpy array of images of shape (m, n_x) 
    m = number of examples
    n_x = number of pixels in an image
    
    """    
    
    imagedata = None
    hd_type = np.dtype('>u4') # unsigned int32 with big endien
    dt_type = np.dtype('>u1') # unsigned int8 with big endient
    
    with open(path + filename) as f:
        # Valid MNIST Label file checker
        magic = np.fromfile(f, dtype=hd_type, count=1)
        if 2051 != magic:
            print('Magic number {} incorrect. Invalid MNIST label file'.format(magic))
        else:
            # Reshape data into 1D array (rows)
            m = int(np.fromfile(f, dtype=hd_type, count=1))
            r = int(np.fromfile(f, dtype=hd_type, count=1))
            c = int(np.fromfile(f, dtype=hd_type, count=1))
            print('Reading {} with {} images of shape: ({}, {})'.format(filename, m, r, c))
            imagedata = np.fromfile(f, dtype=dt_type, count=m*r*c)
            imagedata.shape = (m, r*c)

    return imagedata

def load_dataset():
    
    train_set_x_orig = read_images('./datasets/', 'train-images-idx3-ubyte')
    train_set_y_orig = read_labels('./datasets/', 'train-labels-idx1-ubyte')
    
    test_set_x_orig = read_images('./datasets/', 't10k-images-idx3-ubyte')
    test_set_y_orig = read_labels('./datasets/', 't10k-labels-idx1-ubyte')
  
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
