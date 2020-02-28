from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print(tf.__version__)

import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import time
import cv2


from IPython import display

#mpl.use('Agg')

self_path = os.path.dirname(os.path.abspath(__file__))

class GAN:
    def __init__(self, buff_size=60000, batch_size=256, imgs_size=(28, 28), colors=3, division=4, epochs=60, noise_dim=100):
        self.BUFFER_SIZE = buff_size
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = imgs_size
        self.COLORS = colors
        self.DIVISION = 4
        self.EPOCHS = epochs
        self.noise_dim = noise_dim
        self.num_examples_to_generate = 16
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.checkpoint_path = os.path.join(self_path, 'model_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_path, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer,generator=self.generator, discriminator=self.discriminator,)
        if os.path.exists(self.checkpoint_path):
            self.load_model()
        print("Starting with parameters: ", "BUFFER_SIZE=", self.BUFFER_SIZE, " BATCH_SIZE=", self.BATCH_SIZE, " DATAPART_SIZE=", self.IMAGE_SIZE, " EPOCHS=", self.EPOCHS)
    
    def prepare_dataset(self, dataset):
        dataset = dataset.astype('float32')
        dataset = dataset / 255.0
        return tf.data.Dataset.from_tensor_slices(dataset).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
    
    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(int(self.IMAGE_SIZE[0]/4)*int(self.IMAGE_SIZE[1]/4)*256, use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((int(self.IMAGE_SIZE[0]/4), int(self.IMAGE_SIZE[1]/4), 256)))
        assert model.output_shape == (None, int(self.IMAGE_SIZE[0]/4), int(self.IMAGE_SIZE[1]/4), 256) # Note: None is the batch size

        model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.IMAGE_SIZE[0]/4), int(self.IMAGE_SIZE[1]/4), 128)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.IMAGE_SIZE[0]/2), int(self.IMAGE_SIZE[1]/2), 64)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3)
        
        return model
    
    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=[self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3]))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    
    def generate_image(self):
        noise = tf.random.normal([1, self.noise_dim])
        predictions = self.generator(noise, training=False)
        for i in range(predictions.shape[0]):
            image = np.float32(predictions[i])
            cur_shape = image.shape
            image = cv2.resize(image, (cur_shape[0]*2, cur_shape[1]*2), interpolation=cv2.INTER_LINEAR)
            return image

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(cv2.cvtColor(np.float32(predictions[i] * 255), cv2.COLOR_BGR2RGB).astype(np.uint8))
            plt.axis('off')

        plt.savefig(os.path.join(self_path, 'animation/image_at_epoch_{:04d}.png'.format(epoch)))
        plt.close(fig)
    
    def train(self, dataset):
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        dataset = self.prepare_dataset(dataset)

        for epoch in range(self.EPOCHS):
            start = time.time()
            for image_batch in dataset:
                self.train_step(image_batch)
            
            # Produce images for the GIF as we go
            display.clear_output(wait=True)

            # Save the model every 15 epochs
            if (epoch + 1) % 100 == 0:
                self.save_model()
            if (epoch + 1) % 5 == 0:
                self.generate_and_save_images(self.generator, epoch + 1, seed)
            
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator, epoch + 1, seed)
    
    def save_model(self):
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

    def load_model(self):
        print("Loading weights.")
        latest = tf.train.latest_checkpoint(self.checkpoint_path)
        self.checkpoint.restore(latest)
        self.generator = self.checkpoint.generator
        self.discriminator = self.checkpoint.discriminator
        self.generator_optimizer = self.checkpoint.generator_optimizer
        self.discriminator_optimizer = self.checkpoint.discriminator_optimizer