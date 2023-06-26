
# coding: utf-8

# In[1]:


#from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import cv2
import os
import glob
import keras
import numpy as np
from keras import models
from keras import layers
#from tensorflow import keras
from keras.layers import Dense
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
#import matplotlib.pyplot as pl
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten,  MaxPool2D
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import InceptionV3
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D 
inp=224
#model =  InceptionV3(weights='imagenet',include_top=False,input_shape=(inp, inp, 3))

"""
idx = 0
for layer in model.layers:
    print(idx, layer.name) 
    idx += 1 
    filters = model.layers[idx].get_weights()
    if(len(filters) == 2):
        print(filters[0].shape)
"""

from keras.models import Input
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils import data_utils
from keras.utils import layer_utils

folders = glob.glob("data/*")

img_list = []
label_list=[]

for folder in folders:
    print(folder) 
    for img in glob.glob(folder+r"/*.jpg"):
        #print(img)
        n= cv2.imread(img)
        class_num = folders.index(folder)
        label_list.append(class_num)
        resized = cv2.resize(n, (224,224), interpolation = cv2.INTER_AREA)
        img_list.append(resized)

X_train, X_valid, y_train, y_valid = train_test_split(img_list, label_list, test_size=0.2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1,random_state=1)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
X_test = np.array(X_test)
y_test = np.array(y_test)
print ("training_set", X_train.shape)
print ("training_set", y_train.shape)
print ("validation_set",X_valid.shape)
print ("validation_set",y_valid.shape)
print ("test_set",X_test.shape)
print ("test_set",y_test.shape)
print("Train_Folder",len(X_train))
print("validation_Folder",len(X_valid))
print("Test_Folder",len(X_test))

###############################################

#from . import get_submodules_from_kwargs
#from . import imagenet_utils
#from .imagenet_utils import decode_predictions
#from .imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def InceptionV3_g(include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000, model_path=""):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        pass

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    """
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    """
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')


    # load weights
    if weights == 'imagenet':
        if include_top:
          weights_path = data_utils.get_file(
              'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
              WEIGHTS_PATH,
              cache_subdir='models',
              file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
          weights_path = data_utils.get_file(
              'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
              WEIGHTS_PATH_NO_TOP,
              cache_subdir='models',
              file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


##################################################

def InceptionV3_g_compress(filt1, filt2, filt3, filt4, filt1_1, filt2_1, filt3_1, filt4_1,
                weights=None,
                include_top=False,
                input_shape=None,
                input_tensor=None,
                pooling=None,
                classes=1000, model_path=""):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        pass

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')
    """
    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')
    """

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')
    
    # mixed 9: 8 x 8 x 2048
    #for i in range(2):
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, filt2, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, filt4, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, filt4, 3, 1)
    branch3x3 = layers.concatenate(
        [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(0))

    branch3x3dbl = conv2d_bn(x, filt3, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, filt2, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, filt4, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, filt4, 3, 1)
    branch3x3dbl = layers.concatenate(
        [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed' + str(9))
    
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, filt2_1, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, filt4_1, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, filt4_1, 3, 1)
    branch3x3 = layers.concatenate(
        [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed10_' + str(0))

    branch3x3dbl = conv2d_bn(x, filt3_1, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, filt2_1, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, filt4_1, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, filt4_1, 3, 1)
    branch3x3dbl = layers.concatenate(
        [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed' + str(10))
    
    
    """
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    """
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')


    # load weights
    if weights == 'imagenet':
        if include_top:
          weights_path = data_utils.get_file(
              'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
              WEIGHTS_PATH,
              cache_subdir='models',
              file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
          weights_path = data_utils.get_file(
              'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
              WEIGHTS_PATH_NO_TOP,
              cache_subdir='models',
              file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


################################################

import cv2
import os
import glob
import keras
import numpy as np
#from tensorflow import keras
#import matplotlib.pyplot as pl
model1 =  InceptionV3_g(weights='imagenet',include_top=False,input_shape=(inp, inp, 3))

model1.summary()

from sklearn.model_selection import train_test_split
#inputs1 =Input((inp, inp, 3))

X = model1.output
flat1 = GlobalAveragePooling2D()(X)

#model.summary()
#model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# add new classifier layers
#flat1 = MaxPool2D(pool_size=(2,2),strides=(2,2))(model.layers[-3].output)
x3=layers.GlobalAveragePooling2D()(X)
x1=layers.Dense(100, activation='relu')(flat1)
x1=layers.Dense(50, activation='relu')(x1)
#x1=layers.Dropout(0.25)(x1)
x1=layers.BatchNormalization()(x1)
print("output of channel info ", x1)
#Spatial Attention
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), activation='relu', padding='same')(X)
x2=layers.Conv2D(filters = 64,kernel_size = (3,3), activation='relu', padding='same')(x2)
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), activation='relu', padding='same')(x2)
#x2=layers.Dropout(0.25)(x2)
x2=layers.GlobalAveragePooling2D()(x2)
#x2=layers.Dense(50, activation='relu')(x2)
x2=layers.BatchNormalization()(x2)
print("output of Spatial info ", x2)
##BAM
BAM=layers.concatenate([x1, x2])
BAM=layers.BatchNormalization()(BAM)
print("output of Final BAM ", BAM)

BAM=layers.concatenate([x3, BAM])
F=layers.Dense(150, activation='relu')(BAM)
F=layers.BatchNormalization()(F)
output = Dense(2,activation='softmax')(F)
# define new model
model = Model(inputs=model1.input, outputs=output)


model.summary()



# In[4]:


import datetime
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
batch_size = 8
epochs = 50
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#history=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(X_valid,y_valid))

model_final = model

model_final.load_weights('attn_inceptionv3.h5')

# In[11]:

layer1_filt = 320
layer2_filt = 384
layer3_filt = 448
layer4_filt = 384

layer1_filt_1 = 320
layer2_filt_1 = 384
layer3_filt_1 = 448
layer4_filt_1 = 384

import random


modelc =  InceptionV3_g_compress(layer1_filt, layer2_filt, layer3_filt, layer4_filt, layer1_filt_1, layer2_filt_1, layer3_filt_1, layer4_filt_1, weights=None,include_top=False,input_shape=(inp, inp, 3))

modelc.summary()

from sklearn.model_selection import train_test_split
#inputs1 =Input((inp, inp, 3))

X = modelc.output
flat1 = GlobalAveragePooling2D()(X)

#model.summary()
#model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# add new classifier layers
#flat1 = MaxPool2D(pool_size=(2,2),strides=(2,2))(model.layers[-3].output)
x3=layers.GlobalAveragePooling2D()(X)
x1=layers.Dense(100, activation='relu')(flat1)
x1=layers.Dense(50, activation='relu')(x1)
#x1=layers.Dropout(0.25)(x1)
x1=layers.BatchNormalization()(x1)
print("output of channel info ", x1)
#Spatial Attention
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), activation='relu', padding='same')(X)
x2=layers.Conv2D(filters = 64,kernel_size = (3,3), activation='relu', padding='same')(x2)
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), activation='relu', padding='same')(x2)
#x2=layers.Dropout(0.25)(x2)
x2=layers.GlobalAveragePooling2D()(x2)
#x2=layers.Dense(50, activation='relu')(x2)
x2=layers.BatchNormalization()(x2)
print("output of Spatial info ", x2)
##BAM
BAM=layers.concatenate([x1, x2])
BAM=layers.BatchNormalization()(BAM)
print("output of Final BAM ", BAM)

BAM=layers.concatenate([x3, BAM])
F=layers.Dense(150, activation='relu')(BAM)
F=layers.BatchNormalization()(F)
output = Dense(2,activation='softmax')(F)
# define new model
model1 = Model(inputs=modelc.input, outputs=output)


model1.summary()
"""
B_1 = model_final.layers[243].get_weights()
B_2 = model1.layers[147].get_weights()

print('Len1', len(B_1))
print('Len2', len(B_2))
index1 = 0
# plot each channel separately
for j in range(320):
    if(1 == 1) :
        B_2[0][index1] = B_1[0][j]
        B_2[1][index1] = B_1[1][j]
        B_2[2][index1] = B_1[2][j]
        #B_2[3][index1] = B_1[3][j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[243].set_weights(B_2)
"""
"""
####################### 1st convolution layer with 320 filters
print('1st convolution layer with 320 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

layerr = model_final.layers[241].get_weights()
print(layerr)
filters = model_final.layers[241].get_weights()
filters1 = np.copy(filters)
##biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 320):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters[0])
        ##biases1 = np.copy(biases)
        
        for i in range(0,320):
            f = filters[0][:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[241].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,320):
            f = filters[0][:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[241].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A1 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer1_filt = new_num
"""
####################### 2nd convolution layer with 448 filters
print('2nd convolution layer with 448 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

filters = model_final.layers[249].get_weights()[0]
filters1 = np.copy(filters)
##biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 448):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        #biases1 = np.copy(biases)
        
        for i in range(0,448):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[249].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,448):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[249].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A3 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer3_filt = new_num

####################### 3rd convolution layer with 384 filters
print('3rd convolution layer with 384 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

filters = model_final.layers[252].get_weights()[0]
filters1 = np.copy(filters)
#biases1 = np.copy(biases)
filters_1 = model_final.layers[253].get_weights()[0]
filters1_1 = np.copy(filters_1)
#biases1_1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 384):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        filters1_1 = np.copy(filters_1)
        #biases1 = np.copy(biases)
        
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
     
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        model_final.layers[252].set_weights([filters1])
        model_final.layers[253].set_weights([filters1_1])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        model_final.layers[252].set_weights([filters1])
        model_final.layers[253].set_weights([filters1_1])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A2 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer2_filt = new_num

####################### 4th convolution layer with 384 filters
print('4th convolution layer with 384 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

filters = model_final.layers[258].get_weights()[0]
filters1 = np.copy(filters)
#biases1 = np.copy(biases)
filters_1 = model_final.layers[259].get_weights()[0]
filters1_1 = np.copy(filters_1)
#biases1_1 = np.copy(biases)
filters_2 = model_final.layers[260].get_weights()[0]
filters1_2 = np.copy(filters_2)
#biases1_2 = np.copy(biases)
filters_3 = model_final.layers[261].get_weights()[0]
filters1_3 = np.copy(filters_3)
#biases1_3 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 384):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        filters1_1 = np.copy(filters_1)
        filters1_2 = np.copy(filters_2)
        filters1_3 = np.copy(filters_3)
        #biases1 = np.copy(biases)
        
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
     
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_2[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_2[i] = 0
                filters1_2[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_3[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_3[i] = 0
                filters1_3[:, :, :, i] = 0
        
        model_final.layers[258].set_weights([filters1])
        model_final.layers[259].set_weights([filters1_1])
        model_final.layers[260].set_weights([filters1_2])
        model_final.layers[261].set_weights([filters1_3])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_2[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_2[i] = 0
                filters1_2[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_3[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_3[i] = 0
                filters1_3[:, :, :, i] = 0
        
        model_final.layers[258].set_weights([filters1])
        model_final.layers[259].set_weights([filters1_1])
        model_final.layers[260].set_weights([filters1_2])
        model_final.layers[261].set_weights([filters1_3])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A4 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer4_filt = new_num

########################Next set of filters###############
print('Next set of filters')
"""
####################### 1st convolution layer with 320 filters
print('1st convolution layer with 320 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

filters = model_final.layers[263].get_weights()[0]
filters1 = np.copy(filters)
#biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 320):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        #biases1 = np.copy(biases)
        
        for i in range(0,320):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[263].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,320):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[263].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A1_1 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer1_filt_1 = new_num
"""
####################### 2nd convolution layer with 448 filters
print('2nd convolution layer with 448 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

filters = model_final.layers[280].get_weights()[0]
filters1 = np.copy(filters)
#biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 448):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        #biases1 = np.copy(biases)
        
        for i in range(0,448):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[280].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,448):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[280].set_weights([filters1])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A3_1 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer3_filt_1 = new_num

####################### 3rd convolution layer with 384 filters
print('3rd convolution layer with 384 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

filters = model_final.layers[283].get_weights()[0]
filters1 = np.copy(filters)
#biases1 = np.copy(biases)
filters_1 = model_final.layers[284].get_weights()[0]
filters1_1 = np.copy(filters_1)
#biases1_1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 384):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        filters1_1 = np.copy(filters_1)

        #biases1 = np.copy(biases)
        
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
     
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        model_final.layers[283].set_weights([filters1])
        model_final.layers[284].set_weights([filters1_1])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        model_final.layers[283].set_weights([filters1])
        model_final.layers[284].set_weights([filters1_1])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A2_1 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer2_filt_1 = new_num

####################### 4th convolution layer with 384 filters
print('4th convolution layer with 384 filters')
A = []
Acc = []

arr = model_final.evaluate(X_test,y_test)
print(arr)

filters = model_final.layers[289].get_weights()[0]
filters1 = np.copy(filters)
#biases1 = np.copy(biases)
filters_1 = model_final.layers[290].get_weights()[0]
filters1_1 = np.copy(filters_1)
#biases1_1 = np.copy(biases)
filters_2 = model_final.layers[291].get_weights()[0]
filters1_2 = np.copy(filters_2)
#biases1_2 = np.copy(biases)
filters_3 = model_final.layers[292].get_weights()[0]
filters1_3 = np.copy(filters_3)
#biases1_3 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)
            
    return new_par

popsize = 20
mutate = 0.5
recombination = 0.7
population = []
for i in range(0,popsize):
    indv = []
    for k in range(0, 384):
        indv.append(random.randint(0,1))
    population.append(indv)
        
#--- SOLVE --------------------------------------------+

# cycle through each generation (step #2)
for i in range(1,10+1):
    print ('GENERATION:',i)

    gen_scores = [] # score keeping

    # cycle through each individual in the population
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+
        
        # select three random vector index positions [0, popsize), not including current vector (j)
        canidates = list(range(0,popsize))
        canidates.remove(j)
        random_index = random.sample(canidates, 3)

        x_1 = population[random_index[0]]
        x_2 = population[random_index[1]]
        x_3 = population[random_index[2]]
        x_t = population[j]     # target individual

        # subtract x3 from x2, and create a new vector (x_diff)
        x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

        # multiply x_diff by the mutation factor (F) and add to x_1
        v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
        v_donor = ensure_bounds(v_donor)

        #--- RECOMBINATION (step #3.B) ----------------+

        v_trial = []
        for k in range(len(x_t)):
            crossover = random.random()
            if crossover <= recombination:
                v_trial.append(v_donor[k])

            else:
                v_trial.append(x_t[k])
                
        #--- GREEDY SELECTION (step #3.C) -------------+

        filters1 = np.copy(filters)
        filters1_1 = np.copy(filters_1)
        filters1_2 = np.copy(filters_2)
        filters1_3 = np.copy(filters_3)

        #biases1 = np.copy(biases)
        
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
     
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_2[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_2[i] = 0
                filters1_2[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_3[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                #biases1_3[i] = 0
                filters1_3[:, :, :, i] = 0
        
        model_final.layers[289].set_weights([filters1])
        model_final.layers[290].set_weights([filters1_1])
        model_final.layers[291].set_weights([filters1_2])
        model_final.layers[292].set_weights([filters1_3])
        arr = model_final.evaluate(X_test,y_test)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,384):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_1[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_1[i] = 0
                filters1_1[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_2[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_2[i] = 0
                filters1_2[:, :, :, i] = 0
        
        for i in range(0,384):
            f = filters_3[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                #biases1_3[i] = 0
                filters1_3[:, :, :, i] = 0
        
        model_final.layers[289].set_weights([filters1])
        model_final.layers[290].set_weights([filters1_1])
        model_final.layers[291].set_weights([filters1_2])
        model_final.layers[292].set_weights([filters1_3])
        arr = model_final.evaluate(X_test,y_test)
        score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)


        #score_target = cost_func(x_t)

        if score_trial > score_target:
            population[j] = v_trial
            gen_scores.append(score_trial)
            #print '   >',score_trial, v_trial

        else:
            #print '   >',score_target, x_t
            gen_scores.append(score_target)

    #--- SCORE KEEPING --------------------------------+

    gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
    gen_best = max(gen_scores) 
    print(gen_best)                                 # fitness of best individual
    par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual


A4_1 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer4_filt_1 = new_num


modelc =  InceptionV3_g_compress(layer1_filt, layer2_filt, layer3_filt, layer4_filt, layer1_filt_1, layer2_filt_1, layer3_filt_1, layer4_filt_1, weights=None,include_top=False,input_shape=(inp, inp, 3))

modelc.summary()

from sklearn.model_selection import train_test_split
#inputs1 =Input((inp, inp, 3))

X = modelc.output
flat1 = GlobalAveragePooling2D()(X)

#model.summary()
#model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# add new classifier layers
#flat1 = MaxPool2D(pool_size=(2,2),strides=(2,2))(model.layers[-3].output)
x3=layers.GlobalAveragePooling2D()(X)
x1=layers.Dense(100, activation='relu')(flat1)
x1=layers.Dense(50, activation='relu')(x1)
#x1=layers.Dropout(0.25)(x1)
x1=layers.BatchNormalization()(x1)
print("output of channel info ", x1)
#Spatial Attention
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), activation='relu', padding='same')(X)
x2=layers.Conv2D(filters = 64,kernel_size = (3,3), activation='relu', padding='same')(x2)
x2=layers.Conv2D(filters = 64,kernel_size = (1,1), activation='relu', padding='same')(x2)
#x2=layers.Dropout(0.25)(x2)
x2=layers.GlobalAveragePooling2D()(x2)
#x2=layers.Dense(50, activation='relu')(x2)
x2=layers.BatchNormalization()(x2)
print("output of Spatial info ", x2)
##BAM
BAM=layers.concatenate([x1, x2])
BAM=layers.BatchNormalization()(BAM)
print("output of Final BAM ", BAM)

BAM=layers.concatenate([x3, BAM])
F=layers.Dense(150, activation='relu')(BAM)
F=layers.BatchNormalization()(F)
output = Dense(2,activation='softmax')(F)
# define new model
model1 = Model(inputs=modelc.input, outputs=output)

opt = SGD(lr=0.001, momentum=0.9)
model1.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model1.summary()

model1.save_weights('Fire_incept_pruned_weights.h5')

for k in range(0,326):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    
    if(k >= 133 and k <= 228):
        continue

    layerr = model_final.layers[k].get_weights()
    
    k1 = k
    if(k > 132):
        k1 = k-96
        
    if(k not in [249, 250, 252, 253, 254, 255, 258, 259, 260, 261, 264, 265, 266, 267, 280, 281, 283, 284, 285, 286, 289, 290, 291, 292, 295, 296, 297, 298]):
        try:
            model1.layers[k1].set_weights(layerr)
        except:
            pass
    
    """
    if(k == 241):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(320):
            if(A1[j] == 1) :
                index2 = 0
                for l in range(192):
                    if(1 == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1

        model1.layers[k1].set_weights([filters1])
        
    
    if(k == 243):
        B_1 = model_final.layers[243].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1))
        print('Len2', len(B_2))
        index1 = 0
        # plot each channel separately
        for j in range(320):
            if(A1[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)
    """
    if(k == 249):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(448):
            if(A3[j] == 1) :
                index2 = 0
                for l in range(192):
                    if(1 == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1

        model1.layers[k1].set_weights([filters1])

    if(k == 250):
        B_1 = model_final.layers[250].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1))
        print('Len2', len(B_2))
        index1 = 0
        # plot each channel separately
        for j in range(448):
            if(A3[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)

    if(k == 252 or k == 253):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(384):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(448):
                    if(A3[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1

        model1.layers[k1].set_weights([filters1])

    if(k == 254 or k == 255):
        B_1 = model_final.layers[k].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1))
        print('Len2', len(B_2))
        index1 = 0
        # plot each channel separately
        for j in range(384):
            if(A2[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)


    if(k == 258 or k == 259 or k == 260 or k==261):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(384):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(384):
                    if(A2[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1
        
        model1.layers[k1].set_weights([filters1])

    if(k == 264 or k == 265 or k ==266 or k == 267):
        B_1 = model_final.layers[k].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1))
        print('Len2', len(B_2))
        index1 = 0
        # plot each channel separately
        for j in range(384):
            if(A4[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)

    """
    if(k == 263):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(320):
            if(A1_1[j] == 1) :
                index2 = 0
                for l in range(384):
                    if(A4[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1

        model1.layers[k1].set_weights([filters1])
    if(k == 269):
        B_1 = model_final.layers[269].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1[0]))
        print('Len2', len(B_2[0]))
        index1 = 0
        # plot each channel separately
        for j in range(320):
            if(A1_1[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)
    """
    
    if(k == 280):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(448):
            if(A3_1[j] == 1) :
                index2 = 0
                for l in range(192):
                    if(1 == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1

        model1.layers[k1].set_weights([filters1])

    if(k == 281):
        B_1 = model_final.layers[281].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1))
        print('Len2', len(B_2))
        index1 = 0
        # plot each channel separately
        for j in range(448):
            if(A3_1[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)

    if(k == 283 or k == 284):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(384):
            if(A2_1[j] == 1) :
                index2 = 0
                for l in range(448):
                    if(A3_1[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1

        model1.layers[k1].set_weights([filters1])

    if(k == 285 or k == 286):
        B_1 = model_final.layers[k].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1))
        print('Len2', len(B_2))
        index1 = 0
        # plot each channel separately
        for j in range(384):
            if(A2_1[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)

    if(k == 289 or k == 290 or k == 291 or k == 292):
        filters = model_final.layers[k].get_weights()[0]
        filters1 = model1.layers[k1].get_weights()[0]
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        index1 = 0
        for j in range(384):
            if(A4_1[j] == 1) :
                index2 = 0
                for l in range(384):
                    if(A2_1[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                #biases1[index1] = biases[j]
                index1 += 1

        model1.layers[k1].set_weights([filters1])

    if(k == 295 or k == 296 or k ==297 or k == 298):
        B_1 = model_final.layers[k].get_weights()
        B_2 = model1.layers[k1].get_weights()
        
        print('Len1', len(B_1))
        print('Len2', len(B_2))
        index1 = 0
        # plot each channel separately
        for j in range(384):
            if(A4_1[j] == 1) :
                B_2[0][index1] = B_1[0][j]
                B_2[1][index1] = B_1[1][j]
                B_2[2][index1] = B_1[2][j]
                #B_2[3][index1] = B_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k1].set_weights(B_2)

arr = model1.evaluate(X_test,y_test)
print(arr)

#y_train = y_train.reshape(len(y_train), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_train)


model1.fit(X_train,y_train,batch_size=64,epochs=50)
model1.summary()
model1.save_weights('Fire_incept_pruned_weights.h5')

#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
arr = model1.evaluate(X_test,y_test)

print(arr)
#import matplotlib.pyplot as plt
import seaborn as sn
import seaborn as sns
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import classification_report 
import numpy as np
print(history.history.keys())
#import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))
sn.set(font_scale=1.0)
f, ax = plt.subplots()
ax.plot([None] + history.history['accuracy'])
ax.plot([None] + history.history['val_accuracy'])
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train acc', 'Val acc'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
f, ax = plt.subplots()
ax.plot([None] + history.history['loss'])
ax.plot([None] + history.history['val_loss'])
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train loss', "Val loss"], loc = 1)
ax.set_title('Training/Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()

from sklearn.metrics import  confusion_matrix
#import matplotlib.pyplot as plot
import seaborn as sn
import pandas as pd
import seaborn as sns
import seaborn as sn
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import classification_report 
import numpy as np
y_pred=model.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)

target_names = ["Fire","Non-Fire"] 
cm = confusion_matrix(y_test, y_pred)
print("***** Confusion Matrix *****")
print(cm)
print("***** Classification Report *****")
print(classification_report(y_test, y_pred, target_names=target_names))
classes=2
con = np.zeros((classes,classes))
for x in range(classes):
    for y in range(classes):
        con[x,y] = cm[x,y]/np.sum(cm[x,:])

plt.figure(figsize=(6,4))
sn.set(font_scale=1.5) # for label size
df = sns.heatmap(con, annot=True,fmt='.2', cmap='Blues',xticklabels= target_names , yticklabels= target_names)
df.figure.savefig("InceptionV3.png")
plt.show()
model.save("E:/IMLab/Hikmat Thesis work/paper_01/without_wieght/Dataset 1.h5")
print('\nTesting loss: {:.4f}\nTesting accuracy: {:.4f}'.format(*model.evaluate(X_test, y_test)))

