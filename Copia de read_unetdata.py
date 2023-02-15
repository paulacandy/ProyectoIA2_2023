def recover_bowldata(width =128, height= 128, channels=3, train_path = '../data/bowl-2018/stage1_train/', reduced = True): 
    import os
    import sys
    import random
    import warnings

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from tqdm import tqdm
    from itertools import chain
    from skimage.io import imread, imshow, imread_collection, concatenate_images
    from skimage.transform import resize
    from skimage.morphology import label
    from sklearn.model_selection import train_test_split

    IMG_WIDTH = width # for faster computing on kaggle
    IMG_HEIGHT = height # for faster computing on kaggle
    IMG_CHANNELS = channels
    TRAIN_PATH = train_path
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
   
    train_ids = next(os.walk(TRAIN_PATH))[1]
    
    if reduced is True:
        train_ids = train_ids[:500]
    
    np.random.seed(10)
    
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
        


    print('Todo OK!')
    
    x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    
    return X_train, X_valid, y_train, y_valid 
    

def TGS_data(width =128, height=128, border =5, path = 'data/tgsalt', reduced=True):
    import os
    import random
    import pandas as pd
    import numpy as np
    from tqdm import tqdm_notebook, tnrange
    from itertools import chain
    from skimage.io import imread, imshow, concatenate_images
    from skimage.transform import resize
    from skimage.morphology import label
    from sklearn.model_selection import train_test_split
    im_width = width
    im_height = height
    border = border
    
    
    ids = next(os.walk("images"))[2]
    if reduced is True:
        ids = ids[:500]
    print("total images = ", len(ids))
    
    
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        img = load_img("images/"+id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
        # Load masks
        mask = img_to_array(load_img("masks/"+id_, grayscale=True))
        mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)
        # Save images
        X[n] = x_img/255.0
        y[n] = mask/255.0
        
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
    
    return X_train, X_valid, y_train, y_valid 
