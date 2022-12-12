import matplotlib.pyplot as plt
import os
import tensorflow as tf
import datetime, time
from util.util import is_notebook
from util.image import generate_images
from IPython import display
from tqdm import tqdm


CHECKPOINT_DIR = './training_checkpoints'
LOG_DIR = "logs/"

class Learner:
    
    def __init__(self,
                 generator,
                 discriminator):
        
        self.gen = generator
        self.disc = discriminator
        
        self.set_checkpoint()

        self.log_dir = LOG_DIR
        
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    
    def set_checkpoint(self):
        
        self.checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
        
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer = self.gen.optimizer,
            discriminator_optimizer = self.disc.optimizer,
            generator = self.gen(),
            discriminator = self.disc()
        )
    
    
    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.gen()(input_image, training=True)
            
            disc_real_output = self.disc()(
                [input_image, target], training=True)
            disc_generated_output = self.disc()(
                [input_image, gen_output], training=True
            )
            
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.gen._loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = self.disc._loss(disc_real_output, disc_generated_output)
            
        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.gen().trainable_variables)
        
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.disc().trainable_variables)
        
        self.gen.optimizer.apply_gradients(zip(generator_gradients,
                                                self.gen().trainable_variables))
        
        self.disc.optimizer.apply_gradients(zip(discriminator_gradients,
                                                self.disc().trainable_variables))
        
        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=epoch)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step = epoch)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=epoch)
            tf.summary.scalar("disc_loss", disc_loss, step=epoch)
        

    def fit(self, train_ds, epochs, test_ds, epoch_from = 0, save_only_last=False):
        
        train_size = int(tf.data.experimental.cardinality(train_ds).numpy())
        
        for epoch in range(epoch_from, epochs):
            start = time.time()
            
            if is_notebook():
                display.clear_output(wait=True)

            for example_input, example_target in test_ds.take(1):
                generate_images(self.gen(), example_input, example_target, epoch, save_fig=True)
            print(f"Epoch : {epoch}")
            
            # Train
            for input_image, target in tqdm(train_ds, total=train_size):
                self.train_step(input_image, target, epoch)
            
            # saving (checkpoint) the model every 10 epochs, per approximately 100 minutes
            if ((epoch + 1) % 10 == 0) and (save_only_last == False):
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time()-start))
            
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore(self):
        
        self.checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))