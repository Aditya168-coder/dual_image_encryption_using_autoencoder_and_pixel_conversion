
---

# Autoencoder Model for Image Encryption and Decryption

This repository contains an implementation of an autoencoder model trained on the MNIST dataset for image encryption and decryption.

## Contents

1. [Description](#description)
2. [Usage](#usage)
3. [Files](#files)
4. [Dependencies](#dependencies)
5. [License](#license)

## Description

The autoencoder model is trained on the MNIST dataset to encode and decode images. The encoded representation of the image is used for encryption, and the decoder part of the autoencoder is used to decrypt the encoded image.

## Usage

To encrypt an image:
1. Run the `encryption.py` script, which loads the trained autoencoder model, encodes the input image, converts it to a binary equivalent, and saves the encrypted image to the encryption directory.

To decrypt an image:
1. Run the `decryption.py` script, which loads the trained autoencoder model, loads the encrypted image from the encryption directory, reverses the conversion to obtain the original image, decodes the encrypted image, and saves the decrypted image to the decryption directory.

## Files

- `encrypt_image.py`: Script to encrypt an image using the trained autoencoder model.
- `decrypt_image.py`: Script to decrypt an image using the trained autoencoder model.
- `model/autoencoder_model.h5`: Trained autoencoder model.
- `encryption/`: Directory to store encrypted images.
- `decryption/`: Directory to store decrypted images.

## Dependencies

- Python 3
- TensorFlow
- NumPy
- Matplotlib

Install dependencies using:
```
pip install -r requirements.txt
```

## License

This project is licensed under the [MIT License](LICENSE).

---
