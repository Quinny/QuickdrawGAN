import tensorflow as tf
import time
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image

IMAGE_DIMENSION = 28
LATENT_DIMENSION = 100

# Read all of the images and normalize the gray scale pixels from a range of
# [0, 255] to a range of [-1, 1].
def read_images(file):
    images = np.load(file)
    images = images.reshape((len(images), IMAGE_DIMENSION, IMAGE_DIMENSION))
    images = images.astype(float)
    images = (images - 127.5) / 127.5
    # Trim down to the first 10k examples to make for faster debugging/tuning
    # iterations.
    return images[:1000]

# Transform an image output by a model to a grayscale pillow image object.
def make_image(image):
    image = (image * 127.5) + 127.5
    return Image.fromarray(image.astype('byte'), mode='L')

# Creates the discriminator model used for classifying real images from fake
# ones.
def get_discriminator():
    model = tf.keras.Sequential()

    # Run two rounds of convolutions with increasing number of filters.
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten the output down a single vector and run it through a dense layer
    # with a single classification output.
    #
    # Note that we don't need an activation function here (like one would
    # normally expect to see on a classification output layer) since the cross
    # entropy loss function we use has from_logits set to True.
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Creates the generator model used for learning how to produce "fake" images.
def get_generator():
    model = tf.keras.Sequential()
    # Expand the latent random input into a vector representation of a 7x7
    # 256 channel image and pass it through a dense layer.
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    # None as the first dimension of the shape is the batch size.
    assert model.output_shape == (None, 7, 7, 256)

    # Run the output of the dense layer through a deconvolution to produce the
    # initial "image" representation.
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Run two more deconvolutions which up-sample the image by a factor of 2
    # each time bringing it up to 28x28 to match the "real" images. At the same
    # time, reduce the number of channels down to 1 to make the "real" grayscale
    # image by reducing the number of filters.
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # The final output layer performs a tanh activiation to normalize the
    # output values to (-1, 1).
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False))
    return model

# Generates points in the random latent space.
def generate_latent_points(num_samples):
	return tf.random.normal(shape=(num_samples, LATENT_DIMENSION))

# Initialize all the variables needed for training.
images = read_images("./banana.npy")

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Use a constant seed for generating the progress images so that the "evolution"
# is continuous.
seed = generate_latent_points(10)
generator = get_generator()
discriminator = get_discriminator()

# Display the first image loaded as a sanity check that all the processing
# done was correctly.
make_image(images[0]).show()

# Execute one batch of training.
@tf.function
def train_step(images):
    # Upsample the number of examples in the latent space by 3x to prevent the
    # discriminator from becoming strong too quickly. This ratio was found
    # experimentally as it was observed to work most consistently, your mileage
    # may vary. This may need to be increased for more complicated drawing
    # types.
    noise = generate_latent_points(len(images) * 3)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # Generate fake images using the noise as input.
      generated_images = generator(noise, training=True)

      # Run the discriminator on both the real and fake images.
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      # Compute the loss of the discriminator by counting how accurate it was
      # at classifying both real and fake images.
      real_loss = cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
      disc_loss = real_loss + fake_loss

      # Compute the generator loss by counting how often the discriminator knew
      #the image it produced was fake.
      gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    # Compute and apply the gradients to both models.
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Return some information about the batch to be displayed.
    return gen_loss, disc_loss

# Create the directory to dump the checkpoints and progress images.
if not os.path.isdir("./current"):
    os.makedirs("./current")

# Create the tf checkpoint for saving and loading model progress.
checkpoint_dir = './current/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
)

# Attempt to restore the checkpoint and infer how many epochs have passed
# by counting the checkpoint index files.
try:
    epochs_so_far = len(list(filter(lambda path: path.split(".")[-1] == "index", os.listdir(checkpoint_dir))))
    print("Restoring", tf.train.latest_checkpoint(checkpoint_dir))
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
except Exception as e:
    print(e)
    print("Failed to restore checkpoint")
    epochs_so_far = 0

# Continually run the dataset through the training steps and report the
# aggregate loss after each epoch. Once the discriminator loss starts to get
# close to 0 (somewhere < 0.5) the model has typically converged as much
# as it will.
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        total_gen_loss, total_disc_loss = 0, 0
        total_trained_gen, total_trained_disc = 0, 0
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        # Every 10 epochs run the generator against the fixed seed and save a
        # progress image.
        if epoch % 10 == 0:
            fakes = generator.predict(seed).reshape(28 * len(seed), 28)
            make_image(fakes).save("./current/" + str(epochs_so_far + epoch) + ".png")

        checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epochs_so_far + epoch + 1, time.time() - start))
        print("Gen loss=", total_gen_loss, " total disc loss=", total_disc_loss)


train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(60000).batch(256)
train(train_dataset, 5000)
