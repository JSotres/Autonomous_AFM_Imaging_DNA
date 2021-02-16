
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import cv2
from tensorflow import keras
import os
import logging
import numpy as np

def get_embeddingNetwork(input_shape=(96,96,3), embedding_size=4096):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    base_model = keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_shape=input_shape
    )
    
    inputs = Input(shape=input_shape)
    
    x = base_model(inputs)
    x=Flatten()(x)
    x = Dense(embedding_size, activation='sigmoid',kernel_regularizer=l2(2e-4))(x)
    outputs = Lambda(lambda x: K.l2_normalize(x,axis=-1)) (x)

    embeddingNetwork = Model(inputs, outputs)
    
    return(embeddingNetwork)

def compute_dist(a,b):
    return np.sum(np.square(a-b))

def prepareSavedImageForSiameseEmbedding(image, RESIZE):
    img = cv2.imread(image)
    img = cv2.resize(img, (RESIZE, RESIZE))
    img = (img - img.mean(axis=(0,1)))/img[0].std(axis=(0,1))
    img = img.reshape(-1, RESIZE, RESIZE, 3)
    return img

def prepareNumpyImageForSiameseEmbedding(image, RESIZE):
    img = (image - image.mean(axis=(0,1)))/image.std(axis=(0,1))
    img = img.reshape(-1, RESIZE, RESIZE, 3)
    return img

