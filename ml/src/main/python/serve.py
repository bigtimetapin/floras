import os

import tensorflow.keras as keras
import tensorflow as tf
from preprocess import _decode

DIR = "data/test/"
FILE = "test.jpeg"
DIM_X = 1920
DIM_Y = 1080


def predict(_img, _encoder, _decoder):
    code = _encoder.predict(_img[None])[0]
    return _decoder.predict(code[None])[0]


def write(file_name, img):
    # img = (img + 0.5) * 255.0
    # cv2.imwrite(file_name, img)
    tf.io.write_file(file_name, img)


if __name__ == "__main__":
    # read model
    encoder = keras.models.load_model("data/model/encoder")
    decoder = keras.models.load_model("data/model/decoder")
    # read test data
    img_str = tf.io.read_file(os.path.join(DIR, FILE))
    img = _decode(img_str, DIM_X, DIM_Y)
    # predict
    predicted = predict(img, encoder, decoder)
    write("data/out", predicted)
