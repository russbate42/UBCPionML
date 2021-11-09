
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.run_functions_eagerly(True)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    ''' Without configuration '''
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
        
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2,2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    # def get_config(self):
    #     config = super(TransformerEncoder, self).get_config()
    #     config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
    #     return config

#=============================================================================#
#==================== Simplified Segmentation Class ==========================#
#=============================================================================#

def conv_bn(x, filters):
    ''' Used in the 1D model '''
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    ''' Used in the 1D model '''
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def tnet(inputs, num_features):
    
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)


#=============================================================================#
#==================== Keras Base Segmentation class ==========================#
#=============================================================================#

def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    ''' used in 2d examples'''
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    ''' used in 2d example '''
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)

def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])

class keras_PointNet_segmentation:
    
    def __init__(self, num_points=None, num_features=4, num_classes=3) -> keras.Model:
        
        self.num_points = num_points
        self.num_classes = num_classes
        self.num_features = num_features
        self.model = None
        
    def build(self):
        
        input_points = keras.Input(shape=(None, self.num_features))

        # PointNet Classification Network.
        transformed_inputs = transformation_block(
            input_points, num_features=self.num_features, name="input_transformation_block"
        )
        
        features_32 = conv_block(transformed_inputs, filters=32, name="features_64")
        features_64_1 = conv_block(features_32, filters=64, name="features_128_1")
        features_64_2 = conv_block(features_64_1, filters=64, name="features_128_2")
        
        transformed_features = transformation_block(
            features_64_2, num_features=64, name="transformed_features"
        )
        
        features_256 = conv_block(transformed_features, filters=256, name="features_512")
        features_1024 = conv_block(features_256, filters=1024, name="pre_maxpool_block")
        global_features = layers.MaxPool1D(pool_size=self.num_points, name="global_features")(
            features_1024
        )
        
        global_features = tf.tile(global_features, [1, self.num_points, 1])

        # Segmentation head.
        segmentation_input = layers.Concatenate(name="segmentation_input")(
            [
                features_32,
                features_64_1,
                features_64_2,
                transformed_features,
                features_256,
                global_features,
            ]
        )
        
        segmentation_features = conv_block(
            segmentation_input, filters=64, name="segmentation_features"
        )
        
        outputs = layers.Conv1D(
            self.num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
        )(segmentation_features)
        
        self.model = keras.Model(input_points, outputs)
        return self.model

        
