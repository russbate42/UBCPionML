'''
##============##
## PFN MODELS ##
##============##
- Modeled after the EnergyFlow API

author: Russell Bate
russellbate@phas.ubc.ca
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K


def point_mask_fn(X, mask_val=0.):
    return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))

def tdist_block(x, mask, size: int, number: str):
    dense = layers.Dense(size)
    x = layers.TimeDistributed(dense, name='t_dist_'+number)(x, mask=mask)
    x = layers.Activation('relu', name='activation_'+number)(x)
    return x

def multiply(tensor1, tensor2):
    return K.dot(tensor1,tensor2)

def mat_mul(tensors):
    x, y = tensors
    return tf.linalg.matmul(x,y)

def cast_to_zero(tensors):
    ''' casts all cvalues that should be zero to zero in the modified tensor '''
    mod_input, input_tens = tensors
    full_mask = tf.logical_not(tf.math.equal(input_tens, 0.))
    reduced_mask = tf.experimental.numpy.any(full_mask, axis=-1)
    reduced_mask = tf.cast(reduced_mask, dtype=tf.float32)
    reduced_mask = tf.expand_dims(reduced_mask, axis=-1)
    return_tens = tf.math.multiply(mod_input, reduced_mask)
    return return_tens


## Baseline DNN
def DNN(num_features, name="Baseline - DNN"):
    inputs = keras.Input(shape=(num_features), name='input')
    
    #============== Dumb Layers ==============================================#
    dense_0 = layers.Dense(100)
    t_dist_0 = layers.TimeDistributed(dense_0, name='t_dist_0')(inputs)
    activation_0 = layers.Activation('relu', name="activation_0")(t_dist_0)
    
    dense_1 = layers.Dense(100)
    t_dist_1 = layers.TimeDistributed(dense_1, name='t_dist_1')(activation_0)
    activation_1 = layers.Activation('relu', name='activation_1')(t_dist_1)
    
    dense_2 = layers.Dense(100)
    t_dist_2 = layers.TimeDistributed(dense_2, name='t_dist_2')(activation_1)
    activation_2 = layers.Activation('relu', name='activation_2')(t_dist_2)
    
    dense_3 = layers.Dense(100)
    t_dist_3 = layers.TimeDistributed(dense_3, name='t_dist_3')(activation_2)
    activation_3 = layers.Activation('relu', name='activation_3')(t_dist_3)
    #=========================================================================#
    
    ## Output
    #=========================================================================#
    dense_4 = layers.Dense(1, name='output')(activation_3)
    activation_4 = layers.Activation('linear', name="activation_4")(dense_4)
    #=========================================================================#
    
    return keras.Model(inputs=inputs, outputs=activation_4, name=name)
    
    
## Basic PFN
def PFN_base(num_points, num_features, name="Russell Flow Network"):
    ''' Tested '''
    inputs = keras.Input(shape=(num_points, num_features), name='input')

    #============== Phi Layers ===============================================#
    dense_0 = layers.Dense(100)
    t_dist_0 = layers.TimeDistributed(dense_0, name='t_dist_0')(inputs)
    activation_0 = layers.Activation('relu', name="activation_0")(t_dist_0)
    
    dense_1 = layers.Dense(100)
    t_dist_1 = layers.TimeDistributed(dense_1, name='t_dist_1')(activation_0)
    activation_1 = layers.Activation('relu', name='activation_1')(t_dist_1)
    
    dense_2 = layers.Dense(128)
    t_dist_2 = layers.TimeDistributed(dense_2, name='t_dist_2')(activation_1)
    activation_2 = layers.Activation('relu', name='activation_2')(t_dist_2)
    #=========================================================================#
    
    #============== Aggregation Function (Sum) ===============================#
    lambda_layer = layers.Lambda(point_mask_fn,
                                name='mask')(inputs)

    sum_layer = layers.Dot(axes=(1,1), name='sum')(
        [lambda_layer, activation_2])
    #=========================================================================#
    
    #============== F Layers =================================================#
    dense_3 = layers.Dense(100, name='dense_0')(sum_layer)
    activation_3 = layers.Activation('relu', name="activation_3")(dense_3)
    
    dense_4 = layers.Dense(100, name='dense_1')(activation_3)
    activation_4 = layers.Activation('relu', name="activation_4")(dense_4)
    
    dense_5 = layers.Dense(100, name='dense_2')(activation_4)
    activation_5 = layers.Activation('relu', name="activation_5")(dense_5)
    
    dense_6 = layers.Dense(1, name='output')(activation_5)
    activation_6 = layers.Activation('linear', name="activation_6")(dense_6)
    #=========================================================================#
    
    return keras.Model(inputs=inputs, outputs=activation_6, name=name)


## PFN with more trainable parameters
def PFN_large(num_points, num_features, name="DeepSets - Large"):
    ''' Does more parameters help? More layers? '''
    inputs = keras.Input(shape=(num_points, num_features), name='input')

    #============== Phi Layers ===============================================#
    dense_0 = layers.Dense(200)
    t_dist_0 = layers.TimeDistributed(dense_0, name='t_dist_0')(inputs)
    activation_0 = layers.Activation('relu', name="activation_0")(t_dist_0)
    
    dense_1 = layers.Dense(200)
    t_dist_1 = layers.TimeDistributed(dense_1, name='t_dist_1')(activation_0)
    activation_1 = layers.Activation('relu', name='activation_1')(t_dist_1)
    
    dense_2 = layers.Dense(200)
    t_dist_2 = layers.TimeDistributed(dense_2, name='t_dist_2')(activation_1)
    activation_2 = layers.Activation('relu', name='activation_2')(t_dist_2)
    
    dense_3 = layers.Dense(200)
    t_dist_3 = layers.TimeDistributed(dense_3, name='t_dist_3')(activation_2)
    activation_3 = layers.Activation('relu', name='activation_3')(t_dist_3)
    #=========================================================================#
    
    #============== Aggregation Function (Sum) ===============================#
    lambda_layer = layers.Lambda(point_mask_fn,
                                name='mask')(inputs)

    sum_layer = layers.Dot(axes=(1,1), name='sum')(
        [lambda_layer, activation_3])
    #=========================================================================#
    
    #============== F Layers =================================================#
    dense_4 = layers.Dense(200, name='dense_0')(sum_layer)
    activation_4 = layers.Activation('relu', name="activation_4")(dense_4)
    
    dense_5 = layers.Dense(200, name='dense_1')(activation_4)
    activation_5 = layers.Activation('relu', name="activation_5")(dense_5)
    
    dense_6 = layers.Dense(100, name='dense_2')(activation_5)
    activation_6 = layers.Activation('relu', name="activation_6")(dense_6)
    
    dense_7 = layers.Dense(100, name='dense_3')(activation_6)
    activation_7 = layers.Activation('relu', name="activation_7")(dense_7)
    
    dense_8 = layers.Dense(1, name='output')(activation_7)
    activation_8 = layers.Activation('linear', name="activation_8")(dense_8)
    #=========================================================================#
    
    return keras.Model(inputs=inputs, outputs=activation_8, name=name)


## PFN Adding Dropout on Every Layer!
def PFN_wDropout(num_points, num_features, name="PFN_w_dropout"):
    ''' Tested '''
    inputs = keras.Input(shape=(num_points, num_features), name='input')

    #============== Phi Layers With Dropout ==================================#
    dense_0 = layers.Dense(100)
    t_dist_0 = layers.TimeDistributed(dense_0, name='t_dist_0')(inputs)
    activation_0 = layers.Activation('relu', name="activation_0")(t_dist_0)
    dropout_0 = layers.Dropout(rate=.1, name='dropout_0')(activation_0)
    
    dense_1 = layers.Dense(100)
    t_dist_1 = layers.TimeDistributed(dense_1, name='t_dist_1')(dropout_0)
    activation_1 = layers.Activation('relu', name='activation_1')(t_dist_1)
    dropout_1 = layers.Dropout(rate=.1, name='dropout_1')(activation_1)
    
    dense_2 = layers.Dense(128)
    t_dist_2 = layers.TimeDistributed(dense_2, name='t_dist_2')(dropout_1)
    activation_2 = layers.Activation('relu', name='activation_2')(t_dist_2)
    #=========================================================================#
    
    #============== Aggregration Function (Sum) ==============================#
    lambda_layer = layers.Lambda(point_mask_fn,
                                name='mask')(inputs)

    sum_layer = layers.Dot(axes=(1,1), name='sum')([lambda_layer, activation_2])
    #=========================================================================#
    
    #============== F Layers With Dropout ====================================#
    dense_3 = layers.Dense(100, name='dense_0')(sum_layer)
    activation_3 = layers.Activation('relu', name="activation_3")(dense_3)
    dropout_3 = layers.Dropout(rate=.1, name='dropout_3')(activation_3)
    
    dense_4 = layers.Dense(100, name='dense_1')(dropout_3)
    activation_4 = layers.Activation('relu', name="activation_4")(dense_4)
    dropout_4 = layers.Dropout(rate=.1, name='dropout_4')(activation_4)
    
    dense_5 = layers.Dense(100, name='dense_2')(dropout_4)
    activation_5 = layers.Activation('relu', name="activation_5")(dense_5)
    dropout_5 = layers.Dropout(rate=.1, name='dropout_5')(activation_5)
    
    dense_6 = layers.Dense(1, name='output')(activation_5)
    activation_6 = layers.Activation('linear', name="activation_6")(dense_6)
    #=========================================================================#
    
    return keras.Model(inputs=inputs, outputs=activation_6, name=name)


## PFN with TNet
def PFN_wTNet(num_points, num_features, name="PFN_wTNet"):
    
    inputs = keras.Input(shape=(num_points, num_features), name='input')

    #============== Masking for TNet =========================================#
    mask_tens = layers.Masking(mask_value=0.0, input_shape=(num_points,
                                num_features))(inputs)
    keras_mask = mask_tens._keras_mask
    #=========================================================================#

    #============== TNet =====================================================#
    block_0 = tdist_block(inputs, mask=keras_mask, size=50, number='0')
    block_1 = tdist_block(block_0, mask=keras_mask, size=100, number='1')
    block_2 = tdist_block(block_1, mask=keras_mask, size=100, number='2')
    
    block_2_masked = layers.Lambda(cast_to_zero, name='block_2_masked')(
        [block_2, inputs])
    
    max_pool = layers.MaxPool1D(pool_size=100, padding='valid',
                                name='tnet_0_MaxPool', strides=num_points)(
        block_2_masked)
    
    tnet_0_block_0 = layers.Dense(100, activation='relu',
                                  name='tnet_0_dense_0')(max_pool)
    
    tnet_0_block_1 = layers.Dense(50, activation='relu',
                                  name='tnet_0_dense_1')(tnet_0_block_0)
    
    vector_dense = layers.Dense(
        num_features**2,
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(
            np.eye(num_features).flatten()),
        name='pre_matrix_0'
    )(tnet_0_block_1)
    
    mat_layer = layers.Reshape((num_features, num_features),
                               name='matrix_0')(vector_dense)

    mod_inputs = layers.Lambda(mat_mul, name='matrix_multiply_0')(
        [inputs, mat_layer])
    #=========================================================================#
    
    #============== T_Dist Phi Block =========================================#
    dense_0 = layers.Dense(100)
    t_dist_0 = layers.TimeDistributed(dense_0, name='t_dist_3')(mod_inputs)
    activation_0 = layers.Activation('relu', name="activation_3")(t_dist_0)
    
    dense_1 = layers.Dense(100)
    t_dist_1 = layers.TimeDistributed(dense_1, name='t_dist_4')(activation_0)
    activation_1 = layers.Activation('relu', name='activation_4')(t_dist_1)
    
    dense_2 = layers.Dense(128)
    t_dist_2 = layers.TimeDistributed(dense_2, name='t_dist_5')(activation_1)
    activation_2 = layers.Activation('relu', name='activation_5')(t_dist_2)
    #=========================================================================#
    
    #============== Aggregation Function (Summation) =========================#
    
    # This is important as it produces a layer tensor of 1s and 0s
    # to be dotted with the output of the activation
    lambda_layer = layers.Lambda(point_mask_fn,
                                name='mask')(inputs)

    sum_layer = layers.Dot(axes=(1,1), name='sum')(
        [lambda_layer, activation_2])
    #=========================================================================#

    #============== F Block ==================================================#
    dense_3 = layers.Dense(100, name='dense_6')(sum_layer)
    activation_3 = layers.Activation('relu', name="activation_6")(dense_3)
    
    dense_4 = layers.Dense(100, name='dense_7')(activation_3)
    activation_4 = layers.Activation('relu', name="activation_7")(dense_4)
    
    dense_5 = layers.Dense(100, name='dense_8')(activation_4)
    activation_5 = layers.Activation('relu', name="activation_8")(dense_5)
    
    dense_6 = layers.Dense(1, name='output')(activation_5)
    activation_6 = layers.Activation('linear', name="activation_9")(dense_6)
    #=========================================================================#
    
    return keras.Model(inputs=inputs, outputs=activation_6, name=name)

