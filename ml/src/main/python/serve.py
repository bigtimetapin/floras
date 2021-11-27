import os

import tensorflow.keras as keras
import tensorflow as tf
from preprocess import _decode, DIM_X, DIM_Y

DIR = "data/test/"


def predict(_img, _encoder, _decoder):
    code = _encoder.predict(_img[None])[0]
    return _decoder.predict(code[None])[0]


def write(file_name, _img):
    _img = tf.image.resize(_img, [DIM_X, DIM_Y])
    _img = tf.image.convert_image_dtype(_img, tf.uint8)
    _img_str = tf.image.encode_jpeg(_img)
    tf.io.write_file(file_name, _img_str)


if __name__ == "__main__":
    # read model
    encoder = keras.models.load_model("data/model/encoder")
    decoder = keras.models.load_model("data/model/decoder")
    # read test data
    for file in os.listdir(DIR):
        img_str = tf.io.read_file(os.path.join(DIR, file))
        img = _decode(img_str)
        # predict
        print(img)
        predicted = predict(img, encoder, decoder)
        write("data/out/{0}".format(file), predicted)
