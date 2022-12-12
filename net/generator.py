
from tensorflow.keras.layers import Conv2DTranspose, Dropout, ReLU
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from net.network import Network


class Generator(Network):
    
    def __init__(self,
                 output_channels = 3,
                 _lambda = 100):
        
        super().__init__()
        
        self._lambda = 100
        
        inputs = Input(shape=[256, 256, 3])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64) @7
            self.downsample(128, 4),                        # (bs, 64, 64, 128)  @6
            self.downsample(256, 4),                        # (bs, 32, 32, 256)  @5
            self.downsample(512, 4),                        # (bs, 16, 16, 512)  @4
            self.downsample(512, 4),                        # (bs, 8, 8, 512)    @3
            self.downsample(512, 4),                        # (bs, 4, 4, 512)    @2
            self.downsample(512, 4),                        # (bs, 2, 2, 512)    @1
            self.downsample(512, 4),                        # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)       @1
            self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)       @2
            self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)       @3
            self.upsample(512, 4),                      # (bs, 16, 16, 1024)     @4
            self.upsample(256, 4),                      # (bs, 32, 32, 512)      @5
            self.upsample(128, 4),                      # (bs, 64, 64, 256)      @6
            self.upsample(64, 4),                       # (bs, 128, 128, 128)    @7
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = Conv2DTranspose(output_channels, 4,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer,
                            activation='tanh')  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])    # skip last layer and reverse

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        x = last(x)

        self._model = Model(inputs=inputs, outputs=x)


    def __call__(self):
        return self._model
    
    def _loss(self, disc_generated_output, gen_output, target):
        
        gan_loss = self.loss(
            tf.ones_like(disc_generated_output), disc_generated_output
        )
        
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        
        total_gen_loss = gan_loss + self._lambda * l1_loss
        
        return total_gen_loss, gan_loss, l1_loss