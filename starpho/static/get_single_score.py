import numpy as np
import argparse
from path import Path
import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# from utils.score_utils import mean_score, std_score

def f(x):
    y=np.abs(x-5.5)
    res=np.exp(y)-1;
    return res

def mean_score(scores):
    si = np.arange(1, 11, 1)
    nsi=[f(i) for i in si]
    #print(nsi)
    nsi=np.array(nsi)
    nsi=nsi/nsi.sum()
    vec=scores*nsi;
    vec=vec/vec.sum();
    mean = np.sum(vec*si)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def get_s_score(img_name):
    #parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
    #parser.add_argument('-resize', type=str, default='false',help='Resize images to 224x224 before scoring')

    #args = parser.parse_args()
    #resize_image = args.resize.lower() in ("true", "yes", "t", "1")
    target_size = (224, 224) # if resize_image else None


    with tf.device('/CPU:0'):
        base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights(BASE_DIR + '\static\weights\mobilenet_score_animal.h5')

        img_path=BASE_DIR + r'\api\img\\' + img_name
        #print(1)
        img = load_img(img_path, target_size=target_size)
        #print(2)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred_scores = model.predict(x, batch_size=1, verbose=0)[0]
                #print(np.argmax(pred_scores)+1)
        #print(3)
        mean = mean_score(pred_scores)
        #std = std_score(pred_scores)
        #print(mean)
        #file_name = Path(img_path).name.lower()
        #print(mean)
              
        return mean


