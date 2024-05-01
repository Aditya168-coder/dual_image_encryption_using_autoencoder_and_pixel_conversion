import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # USED TO SUPPRESS UNNECESSARY WARNING FROM TENSORFLOW
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

def encoder_function(image):
    encoded_image = (autoencoder.get_layer('reshape')(autoencoder.get_layer('leaky_re_lu_2')(autoencoder.get_layer('dense')(autoencoder.get_layer('Latent')(autoencoder.get_layer('flatten')(autoencoder.get_layer('max_pooling2d_1')(autoencoder.get_layer('leaky_re_lu_1')(autoencoder.get_layer('batch_normalization_1')(autoencoder.get_layer('conv2d_1')(autoencoder.get_layer('max_pooling2d')(autoencoder.get_layer('leaky_re_lu')(autoencoder.get_layer('batch_normalization')(autoencoder.get_layer('conv2d')(autoencoder.get_layer('input_1')(image))))))))))))))).numpy()
    encoded_image = np.resize(encoded_image, (7, 7))
    integer_array = np.round(encoded_image).astype(int)
    return integer_array

def convert_to_binary_equivalent(image):
    binary_equivalent_image = []

    for row in image:
        binary_row = []
        for pixel_value in row:
            # Handle negative values using two's complement
            if pixel_value < 0:
                pixel_value += 256
            # Convert pixel value to 8-bit binary
            binary_value = format(pixel_value, '010b')
            # Replace pairs of two bits
            modified_value = ''
            for i in range(0, len(binary_value), 2):
                pair = binary_value[i:i+2]
                if pair == '00':
                    modified_value += 'A'
                elif pair == '01':
                    modified_value += 'B'
                elif pair == '10':
                    modified_value += 'C'
                elif pair == '11':
                    modified_value += 'D'
            binary_row.append(modified_value)
        binary_equivalent_image.append(binary_row)

    return binary_equivalent_image

if __name__ == "__main__":
    autoencoder = load_model("model/autoencoder_model.h5")
    dataset = tf.keras.datasets.mnist
    (x_train, _), (x_test, _) = dataset.load_data()

    # Reshape the data to have four dimensions
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    sample_image = x_test[0]
    sample_image = sample_image.reshape(-1, 28, 28, 1)
    encoded_sample = encoder_function(sample_image)
    image = convert_to_binary_equivalent(encoded_sample)

    # Save the image to the "encryption" directory
    encryption_dir = "encryption"
    if not os.path.exists(encryption_dir):
        os.makedirs(encryption_dir)

    np.savetxt('encryption/binary_equivalent_image.txt', image, fmt='%s')

