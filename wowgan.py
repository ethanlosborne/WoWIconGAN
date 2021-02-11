#setup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown
import cv2
from zipfile import ZipFile


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



# create dataset and resize
datasetlowerres = keras.preprocessing.image_dataset_from_directory("wowiconslowres", label_mode=None, image_size=(64, 64), batch_size=32, shuffle=True, seed=1, subset='training', validation_split = 0.1)

datasetlowres = keras.preprocessing.image_dataset_from_directory("wowiconslowres", label_mode=None, image_size=(64, 64), batch_size=32, shuffle=True, seed=1, subset='training', validation_split = 0.1)

dataset = keras.preprocessing.image_dataset_from_directory("wowicons", label_mode=None, image_size=(64, 64), batch_size=32, shuffle=True, seed=1, subset='training', validation_split = 0.1)


# Normalize the images to [-1, 1]
datasetlowerres = datasetlowerres.map(lambda x: (x - 127.5) / 127.5)
datasetlowres = datasetlowres.map(lambda x: (x - 127.5) / 127.5)
dataset = dataset.map(lambda x: (x - 127.5) / 127.5)


# create discriminator
discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()


# create generator
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
    ],
    name="generator",
)
generator.summary()



# override train step
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)



        def smooth_labels(labels, factor=0.1):
            # smooth the labels
            labels *= (1 - factor)
            labels += (factor / labels.shape[1])
            # returned the smoothed labels
            return labels

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0

        )


        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))


        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

# Create a callback that periodically saves generated images
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=100, iteration="low"):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.iteration = iteration

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(self.iteration + "_generated_img_%03d_%d.png" % (epoch, i))


# Train the end-to-end model
epochs = 200  # In practice, use ~100 epochs # use 10-20 for testing

print("creating gan now")
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
)




checkpointpath = "savedcheckpoint"


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointpath,
                                                 save_weights_only=True,
                                                 verbose=1)


print("done, starting to fit")
# loads the checkpoint. Comment out if you want a clean start
gan.load_weights(checkpointpath)


# These two fits are for running the GAN on low and medium res images. Only for first training.
#gan.fit(
#    datasetlowerres, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim, iteration="low"), cp_callback]
#)

#print("DONE WITH FIRST RUNTHROUGH, doing it again")

#gan.fit(
#    datasetlowres, epochs=epochs, callbacks=[GANMonitor(num_img=5, latent_dim=latent_dim, iteration="med"), cp_callback]
#)

#print("DONE WITH SECOND RUNTHROUGH, doing it again")

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=5, latent_dim=latent_dim, iteration="high"), cp_callback]
)
