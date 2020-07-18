import keras
import tensorflow as tf
import tensorflow.compat.v2 as tf_v2
from keras import backend as K
from keras.layers.core import Reshape, Lambda, Dense, Activation
from keras.layers import BatchNormalization, Input
from keras.layers.convolutional import  Conv2D, MaxPooling2D
from keras.layers import  Bidirectional
from keras.layers.merge import average, concatenate
from keras.layers.recurrent import LSTM
from keras.optimizers import Adadelta
from keras.models import Model
import numpy as np
import cv2

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#char.txt consist of all unicode characters in hindi and file should be present in the working directory
with open('char.txt','r',encoding = 'utf-8') as f:
    unicodes = [x.strip() for x in f.readlines()]
f.close()

def preprocess(imgBW,sess):
    """
    Grayscale image must be passed into the function.
    """
    #Otsu Thresholding
    _,im = cv2.threshold(imgBW,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Reshaping as batchsize, height, width, no of channels
    im = im.reshape(1,im.shape[0],im.shape[1],1)
    #Conversion into tensor
    im = tf.convert_to_tensor(im)
    #Using lanczos3 interpolation for resizing the image
    im = tf_v2.image.resize(im,[32,128],method = 'lanczos5',antialias=True)
    #Conversion of tensor into numpy array
    im = im.eval(session=sess)
    im = im.reshape(32,128)
    #conversion of float32 into uint8 datatype
    img_n = cv2.normalize(src=im, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #Otsu thersholding
    _,im = cv2.threshold(img_n,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im = im.astype('float32')
    #Normalization of image
    im /= 255
    (m, s) = cv2.meanStdDev(im)
    m = m[0][0]
    s = s[0][0]
    img = im - m
    img = img / s if s>0 else img
    return im

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True, pad = 'same' ):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding= pad,
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def res_block(inputs,num_filters=16,kernel_size=3,strides=1,padding = 'same',activation='relu',batch_normalization=True,conv_first=True,BN=True,A=True):
    x = inputs
    y = resnet_layer(inputs=x,num_filters=num_filters,strides=strides, pad = padding)
    y = resnet_layer(inputs=y,num_filters=num_filters,activation=None, pad = padding)
    x = resnet_layer(inputs=x,num_filters=num_filters,strides=strides,pad = padding,activation=None,batch_normalization=False)
    x = keras.layers.add([x, y])
    if BN:
        x = BatchNormalization()(x)
    if A:
        x = Activation('relu')(x)
    return x

def getModelCRNN(training):
    inputs = Input(name = 'inputX', shape=(32,128,1), dtype = 'float32')
    inner = res_block(inputs,64)
    inner = res_block(inner,64)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName1')(inner)
    inner = res_block(inner,128)
    inner = res_block(inner,128)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName2')(inner)
    inner = res_block(inner,256)
    inner = res_block(inner,256)
    inner = MaxPooling2D(pool_size = (1,2),strides = (2,2), name = 'MaxPoolName4')(inner)
    inner = res_block(inner,512)
    inner = res_block(inner,512)
    inner = MaxPooling2D(pool_size = (1,2), strides = (2,2), name = 'MaxPoolName6')(inner)
    inner = res_block(inner,512)
    inner = Reshape(target_shape = (32,256), name = 'reshape')(inner)
    blstm1 = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer = 'he_normal'))(inner)
    blstm1 = BatchNormalization()(blstm1)
    blstm2 = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer = 'he_normal'))(blstm1)
    blstm2 = BatchNormalization()(blstm2)
    yPred = Dense(len(unicodes)+1, kernel_initializer = 'he_normal', activation = 'softmax')(blstm2)
    Model(inputs = inputs, outputs = yPred)

    labels = Input(name='label', shape=[32], dtype='float32')
    inputLength = Input(name='inputLen', shape=[1], dtype='int64')
    labelLength = Input(name='labelLen', shape=[1], dtype='int64')

    lossOut = Lambda(ctcLambdaFunc, output_shape=(1,), name='ctc')([yPred, labels, inputLength, labelLength])

    if training:
        return Model(inputs = [inputs, labels, inputLength, labelLength], outputs=[lossOut,yPred])
    return Model(inputs=[inputs], outputs=yPred)


def decode(yPred):#Beam Search
    texts = []
    for i in range(yPred.shape[0]):
        y = yPred[i,2:,:]
        y = np.reshape(y,(1,30,130))
        pred =  K.get_value(K.ctc_decode(y, input_length=np.ones(y.shape[0])*30, greedy=False, beam_width=3, top_paths=1)[0][0])[0]
        word = ""
        for i in range(len(pred)):
            if pred[i] == len(unicodes):
                word+= ""
            else:
                word += unicodes[pred[i]]
        texts.append(word)
    return texts

def labelsToText(labels):
    ret = []
    for c in labels:
        if c == len(unicodes):
            ret.append("")
        else:
            ret.append(unicodes[c])
    return "".join(ret)

def ctcLambdaFunc(args):
    yPred, labels, inputLength, labelLength = args
    yPred = yPred[:,2:,:]
    loss = K.ctc_batch_cost(labels,yPred,inputLength,labelLength)
    return loss

# Word consists of coordinates x1,x2,,y1,y2 => top,bottom, left, right boundaries
def linesort(words):
    X2 = words[:,1]
    index = np.argsort(X2)
    mat = X2[index]

    i = start = ln = 0
    end = -1
    line = {}

    while i <= len(mat)-2:
        if(mat[i+1]-mat[i] > 20):
            start,end = end + 1,i
            line[ln] = index[start:end+1]
            ln +=1
            if(i == len(mat)-2):
                line[ln] = index[i+1]
        elif(mat[i+1]-mat[i] <= 20) & (i == len(mat)-2):
            line[ln] = index[end+1:]
        i +=1
    return line

def wordsort(words,lineindex):
    if np.isscalar(lineindex):
        out = words[lineindex]
    else:
        wordindex = np.argsort(words[lineindex,3])
        out = words[np.array(lineindex)[wordindex]]
    return out

def predict(img,model,coordinates,sess):
    x1,x2,y1,y2 = coordinates
    img = img[x1:x2,y1:y2]
    img = preprocess(img,sess)
    img = np.reshape(img,(1,img.shape[0],img.shape[1],1))
    out = model.predict(img)
    pred = decode(out)
    for word in pred:
        return str(word)
