import numpy as np
import tensorflow as tf
from keras import layers
from keras.constraints import unit_norm, max_norm
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class Expand(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.expand_dims(X, X.ndim)


class Squeeze(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = np.squeeze(X.copy())
        # print(np.shape(X_))
        return X_


def basic_DNN(
    num_chans=32, samples=256, num_hidden=16, activation="relu", learning_rate=1e-3
):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(num_chans, samples, 1)))
    model.add(layers.Dense(num_hidden, activation=activation))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    return model


def basic_DNN2d(input_shape=32, num_hidden=16, activation="relu", learning_rate=1e-3):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(num_hidden, activation=activation))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )

    return model


def DNN(num_chans=32, samples=256, activation="relu", learning_rate=1e-3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(num_chans, samples, 1)))
    model.add(layers.Dense(60, activation=activation))
    model.add(layers.Dense(30, activation=activation))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )

    return model


def DNNb(num_chans=32, samples=256, activation="relu", learning_rate=1e-3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(num_chans, samples, 1)))
    model.add(layers.Dense(32, activation=activation))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(layers.Dense(32, activation=activation))
    model.add(tf.keras.layers.Dropout(0.35))
    model.add(layers.Dense(16, activation=activation))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )

    return model


def DNNb_2d(input_shape=32, activation="relu", learning_rate=1e-3):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(32, activation=activation))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(layers.Dense(32, activation=activation))
    model.add(tf.keras.layers.Dropout(0.35))
    model.add(layers.Dense(16, activation=activation))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )

    return model


def SCNNa(num_chans=32, samples=256, learning_rate=1e-3):
    """
    Shallow CNN
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=50,
            kernel_size=(25, 1),
            padding="same",
            activation="elu",
            input_shape=(num_chans, samples, 1),
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 1), strides=(3, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    return model


# need these for ShallowConvNet
def square(x):
    return tf.keras.backend.square(x)


def log(x):
    return tf.keras.backend.log(
        tf.keras.backend.clip(x, min_value=1e-7, max_value=10000)
    )


def SCNNb(num_chans=32, samples=256, learning_rate=1e-3, dropout_rate=0.5):
    """
    Structure from Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping. Original code :
    https://github.com/vlawhern/arl-eegmodels
    """
    # start the model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            40,
            (1, 25),
            input_shape=(num_chans, samples, 1),
            padding="same",
            kernel_constraint=tf.keras.constraints.max_norm(2.0, axis=(0, 1, 2)),
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            40,
            (num_chans, 1),
            use_bias=False,
            kernel_constraint=tf.keras.constraints.max_norm(2.0, axis=(0, 1, 2)),
        )
    )
    model.add(tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
    model.add(tf.keras.layers.Activation(square))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(1, 75), strides=(1, 15)))
    model.add(tf.keras.layers.Activation(log))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(
            1,
            kernel_constraint=tf.keras.constraints.max_norm(0.5),
            activation="sigmoid",
        )
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )
    return model


def eegnet(
    num_chans=32,
    samples=256,
    kern_length=64,
    dropout_rate=0.5,
    F1=8,
    D=2,
    F2=16,
    P1=4,
    learning_rate=1e-3,
):
    """
    EEGNet inspired by this code:
    https://github.com/vlawhern/arl-eegmodels which is the Keras
    implementation of : http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    :param num_chans: Number of channels
    :param samples: Number of sample
    :param dropout_rate: Dropout fraction
    :param sampling_rate: Sampling rate
    :param F1: Number of temporal filters (F1) to learn.
               Number of pointwise filters F2 = F1 * D.
    :param D: Number of spatial filters to learn within each temporal convolution.
    :param P1: width of the input windows for the average pooling.
    :param learning_rate: Learning rate.
    :return: model
    """
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            F1,
            (1, kern_length),
            padding="same",
            input_shape=(num_chans, samples, 1),
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(
        layers.DepthwiseConv2D(
            (num_chans, 1),
            padding="valid",
            depth_multiplier=D,
            depthwise_constraint=unit_norm(),
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("elu"))
    model.add(layers.AveragePooling2D(pool_size=(1, P1), padding="valid"))
    model.add(layers.Dropout(dropout_rate))
    model.add(
        layers.SeparableConv2D(
            F2, (1, int(kern_length / 2)), use_bias=False, padding="same"
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("elu"))
    P2 = int(P1 * 2)
    model.add(layers.AveragePooling2D(pool_size=(1, P2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, kernel_constraint=max_norm(0.25)))
    model.add(layers.Activation("sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )

    return model


def cust_eegnet(
    num_chans=32,
    samples=256,
    dropout_rate=0.5,
    sampling_rate=512,
    F1=8,
    D=2,
    P1=4,
    learning_rate=1e-3,
):
    """
    EEGNet inspired by this code:
    https://github.com/vlawhern/arl-eegmodels which is the Keras
    implementation of : http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    :param num_chans: Number of channels
    :param samples: Number of sample
    :param dropout_rate: Dropout fraction
    :param sampling_rate: Sampling rate
    :param F1: Number of temporal filters (F1) to learn.
               Number of pointwise filters F2 = F1 * D.
    :param D: Number of spatial filters to learn within each temporal convolution.
    :param P1: width of the input windows for the average pooling.
    :param learning_rate: Learning rate.
    :return: model
    """
    model = tf.keras.Sequential()
    kern_length = int(sampling_rate / 2)
    model.add(
        layers.Conv2D(
            F1,
            (1, kern_length),
            padding="same",
            input_shape=(num_chans, samples, 1),
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(
        layers.DepthwiseConv2D(
            (num_chans, 1),
            padding="valid",
            depth_multiplier=D,
            depthwise_constraint=unit_norm(),
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("elu"))
    model.add(layers.AveragePooling2D(pool_size=(1, P1), padding="valid"))
    model.add(layers.Dropout(dropout_rate))
    F2 = F1 * D
    model.add(
        layers.SeparableConv2D(
            F2, (1, int(kern_length / 2)), use_bias=False, padding="same"
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("elu"))
    P2 = int(P1 * 2)
    model.add(layers.AveragePooling2D(pool_size=(1, P2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, kernel_constraint=max_norm(0.25)))
    model.add(layers.Activation("sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"]
    )

    return model
