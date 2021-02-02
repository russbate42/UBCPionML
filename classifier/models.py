import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor

# Custom layers.
from layers import *

# Our baseline, fully-connected neural network for classification.
# Operates on a single vector (e.g. flattened image from one calorimeter layer).
# Optionally uses dropouts between layers.
def baseline_nn_model(strategy, lr=5e-5, dropout=-1.):
    # create model
    def mod(number_pixels):
        with strategy.scope():    
            model = Sequential()
            used_pixels = number_pixels
            model.add(Dense(number_pixels, input_dim=number_pixels, kernel_initializer='normal', activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))
            model.add(Dense(used_pixels, activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))
            model.add(Dense(int(used_pixels/2), activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))
            model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
            # compile model
            optimizer = Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    return mod

# A simple implementation of ResNet.
# As input, this takes multiple images, which may be of different sizes,
# and they are all rescaled to a user-specified size.
def resnet(strategy, channels=6, lr=5e-5, classes=2):
    # create model
    def mod(input_shape):
        with strategy.scope():
            
            # Input images -- one for each channel, each channel's dimensions may be different.
            inputs = [Input((None,None,1),name='input'+str(i)) for i in range(channels)]
            
            # Rescale all the input images, so that their dimensions now match.
            scaled_inputs = [tf.image.resize(x,input_shape,name='scaled_input'+str(i)) for i,x in enumerate(inputs)]
            
            # Now "stack" the images along the channels dimension.
            X = tf.concat(values=scaled_inputs, axis=3, name='concat')
            #print('In:',X.shape)
            
            X = ZeroPadding2D((3,3))(X)
            #print('S0:',X.shape)
            
            # Stage 1
            X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name='bn_conv1')(X)
            X = Activation('relu')(X)
            X = MaxPooling2D((3, 3), strides=(2, 2))(X)
            #print('S1:',X.shape)
            
            # Stage 2
            filters = [64, 64, 256]
            f = 3
            stage = 2
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=1)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            #print('S2:',X.shape)
            
            # Stage 3
            filters = [128, 128, 512]
            f = 3
            stage = 3
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            X = identity_block(X, f, filters, stage=stage, block='d')
            #print('S3:',X.shape)

            # Stage 4
            filters = [256, 256, 1024]
            f = 3
            stage = 4
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            X = identity_block(X, f, filters, stage=stage, block='d')
            X = identity_block(X, f, filters, stage=stage, block='e')
            X = identity_block(X, f, filters, stage=stage, block='f')
            #print('S4:',X.shape)

            # Stage 5
            filters = [512, 512, 2048]
            f = 3
            stage = 5
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            #print('S5:',X.shape)

            # AVGPOOL
            pool_size = (2,2)
            if(X.shape[1] == 1):   pool_size = (1,2)
            elif(X.shape[2] == 1): pool_size = (2,1)
            X = AveragePooling2D(pool_size=pool_size, name="avg_pool")(X)
            #print('S6:',X.shape)

            # output layer
            X = Flatten()(X)
            #print('S7:',X.shape)
            
            X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
            # Create model object.
            model = Model(inputs=inputs, outputs=X, name='ResNet50')
        
            # Compile the model
            optimizer = Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    return mod

# A simple combination network -- in practice we may use this to
# "combine" classification scores from single calo-layer networks.
def simple_combine_model(strategy, lr=1e-3):
    # create model
    def mod(n_input=6):
        with strategy.scope():
            model = Sequential()
            model.add(Dense(n_input, input_dim=n_input, kernel_initializer='normal', activation='relu'))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
            # compile model
            optimizer = Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    return mod