import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # USED TO SUPPRESS UNNECESSARY WARNING FROM TENSORFLOW
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, Flatten, Reshape, Conv2DTranspose, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def train_autoencoder():
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset = tf.keras.datasets.mnist
    (x_train, _), (x_test, _) = dataset.load_data()

    # Reshape the data to have four dimensions
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    H = 28
    W = 28
    C = 1
    latent_dim = 128
    lr = 1e-3
    batch_size = 32
    epochs = 5

    # model
    inputs = Input(shape=(H, W, C))
    x = inputs

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.2)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.2)(x)
    x = MaxPool2D((2, 2))(x)

    x = Flatten()(x)
    units = x.shape[1]
    x = Dense(latent_dim, name='Latent')(x)
    x = Dense(7*7*1)(x)  # Adjust the number of units for reshape
    x = LeakyReLU(alpha=.2)(x)
    comp = Reshape((7, 7, 1))(x)  # Reshape to (7, 7, 1)

    x = Conv2DTranspose(32, (4, 4), strides=2, padding='same')(comp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.2)(x)

    x = Conv2DTranspose(1, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    outputs = x

    autoencoder = Model(inputs, outputs, name="conv_autoencoder")
    autoencoder.compile(optimizer=Adam(lr), loss='binary_crossentropy')
    autoencoder.summary()

    # training
    history = autoencoder.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test)
    )

    # Create a folder named "model" if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')

    # Save the model to the "model" folder
    autoencoder.save('model/autoencoder_model.h5')

if __name__ == "__main__":
    train_autoencoder()
