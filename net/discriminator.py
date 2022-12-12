from tensorflow.keras.layers import Input, Concatenate
from net.network import Network
from tensorflow.keras.layers import ZeroPadding2D
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras import Model

class Discriminator(Network):
    
    def __init__(self):
        
        super().__init__()
        
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = Input(shape=[256, 256, 3], name='input_image')
        tar = Input(shape=[256, 256, 3], name='target_image')

        x = Concatenate()([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = Conv2D(512, 4, strides=1,
                        kernel_initializer=initializer,
                        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = BatchNormalization()(conv)

        leaky_relu = LeakyReLU()(batchnorm1)

        zero_pad2 = ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = Conv2D(1, 4, strides=1,
                        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        self._model = Model(inputs=[inp, tar], outputs=last)
        
    def __call__(self):
        return self._model
    
    def _loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss(
            tf.ones_like(disc_real_output), disc_real_output
        )
        generated_loss = self.loss(tf.zeros_like(
            disc_generated_output), disc_generated_output
        )
        
        total_dis_loss = real_loss + generated_loss
        
        return total_dis_loss