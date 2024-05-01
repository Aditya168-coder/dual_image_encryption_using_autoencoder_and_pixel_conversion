import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # USED TO SUPPRESS UNNECESSARY WARNING FROM TENSORFLOW
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

def reverse_conversion(binary_equivalent_image):
    original_image = []

    for row in binary_equivalent_image:
        original_row = []
        for binary_value in row:
            # Replace each character with corresponding binary value
            original_value = ''
            for char in binary_value:
                if char == 'A':
                    original_value += '00'
                elif char == 'B':
                    original_value += '01'
                elif char == 'C':
                    original_value += '10'
                elif char == 'D':
                    original_value += '11'
            # Convert binary value to integer
            original_value = int(original_value, 2)
            # Convert back to negative if needed
            if original_value >= 128:
                original_value -= 256
            original_row.append(original_value)
        original_image.append(original_row)

    return np.array(original_image)

def decoder_function(encoded_image):
    # Reshape the encoded image to match the expected input shape of the Latent layer
    reshaped_encoded_image = tf.reshape(encoded_image, (-1,7,7,1))

    # Use the decoder part of the autoencoder to reproduce the image from the low-dimensional representation
    reconstructed_image = (autoencoder.get_layer('activation')(autoencoder.get_layer('batch_normalization_3')(autoencoder.get_layer('conv2d_transpose_1')(autoencoder.get_layer('leaky_re_lu_3')(autoencoder.get_layer('batch_normalization_2')(autoencoder.get_layer('conv2d_transpose')(reshaped_encoded_image))))))).numpy()
    return reconstructed_image

if __name__ == "__main__":
    # Load the autoencoder model
    autoencoder = load_model("model/autoencoder_model.h5")

    # Load the encrypted image from the encryption directory
    saved_binary_equivalent_image = np.loadtxt('encryption/binary_equivalent_image.txt', dtype=str)


    # Reverse the conversion to obtain the original image
    reversed_image = reverse_conversion(saved_binary_equivalent_image)
    reversed_image = reversed_image / 1.0
    encoded_image = tf.reshape(reversed_image, (-1,7,7,1))

    # Decode the encrypted image to obtain the decrypted image
    decoded_image = decoder_function(encoded_image)

    # Save the decrypted image to the decryption folder
    decryption_dir = "decryption"
    if not os.path.exists(decryption_dir):
        os.makedirs(decryption_dir)

    plt.imshow(decoded_image[0], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(decryption_dir, "decrypted_image.png"), bbox_inches='tight', pad_inches=0)
