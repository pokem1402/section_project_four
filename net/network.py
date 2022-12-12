from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Dropout, ReLU
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy

class Network:
    
    def __init__(self):
        self.set_optimizer()
        self._loss_ = BinaryCrossentropy(from_logits=True)
    
    @property
    def loss(self):
        return self._loss_
    
    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        layer = Sequential()

        layer.add(Conv2D(filters, size, strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False))

        if apply_batchnorm:
            layer.add(BatchNormalization())

        layer.add(LeakyReLU())
        return layer


    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        layer = tf.keras.Sequential()
        layer.add(
            Conv2DTranspose(filters, size, strides=2,
                            padding='same',
                            kernel_initializer=initializer,
                            use_bias=False))

        layer.add(BatchNormalization())

        if apply_dropout:
            layer.add(Dropout(0.5))

        layer.add(ReLU())

        return layer   
    
    def set_optimizer(self, learning_rate = 2e-4, beta_1 = 0.5):
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1 = beta_1,
        )
            
    @property
    def optimizer(self):
        return self._optimizer