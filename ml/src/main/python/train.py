import os

import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
import numpy as np
from sklearn.model_selection import train_test_split

DIR = "data/in/"
RESIZE_FACTOR = 4  # 4
DIM_X = int(1920 / RESIZE_FACTOR)
DIM_Y = int(1080 / RESIZE_FACTOR)
COMPRESSION_FACTOR = int(32 * 5)  # * 5)

image_feature_description = {
    'img_str': tf.io.FixedLenFeature([], tf.string),
}

def _decode(img_str):
    # img_str = proto['img_str']
    img = tf.image.decode_jpeg(img_str)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [DIM_X, DIM_Y])
    return img


def decode(proto):
    img = tf.py_function(
        _decode,
        [proto['img_str']],
        tf.float32
    )
    return img


def _dataset():
    # encode as dataset
    dataset = tf.data.TFRecordDataset('data/images.tfrecords')
    # parse
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, image_feature_description))
    ## # decode
    ## dataset = dataset.map(lambda x: decode(x))
    ## # tup
    ## dataset = dataset.map(lambda x: (x, x))
    # batch
    return dataset

"""
TODO: try this stuff
stacked_ae = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
    keras.layers.Dense(100, activaon="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stacked_ae.compile(loss="binary_crossentropy",
                   optimizer=keras.optimizers.SGD(lr=1.5))

history = stacked_ae.fit(img_train, img_train, epochs=10,
                         validation_data=(img_test, img_test))
"""


def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(input_shape=img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))
    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer(input_shape=(code_size,)))
    decoder.add(Dense(np.prod(img_shape)))
    decoder.add(Reshape(img_shape))
    return encoder, decoder


def predict(img, encoder, decoder):
    code = encoder.predict(img[None])[0]
    return decoder.predict(code[None])[0]


def write(file_name, img):
    # img = (img + 0.5) * 255.0
    # cv2.imwrite(file_name, img)
    tf.io.write_file(file_name, img)


if __name__ == "__main__":
    tf.autograph.set_verbosity(10)
    # read photos
    print("reading data . . ")
    dataset = _dataset()
    dataset = dataset.map(lambda x: decode(x))
    # # split train,test
    # print("split train, test")
    # train, test = train_test_split(data, test_size=0.1, random_state=42)
    # print("to dataset")
    # X_train = to_dataset(train)
    # X_test = to_dataset(test)
    # # get dimensions
    # print("get dimensions")
    # IMG_SHAPE = data.shape[1:]
    # # build auto encoder
    # print("build auto encoder")
    # encoder, decoder = build_autoencoder(IMG_SHAPE, COMPRESSION_FACTOR)
    # print("inp")
    # inp = Input(IMG_SHAPE)
    # print("code")
    # code = encoder(inp)
    # print("reconstruction")
    # reconstruction = decoder(code)
    # print("autoencoder")
    # autoencoder = Model(inp, reconstruction)
    # print("compile")
    # autoencoder.compile(optimizer='adamax', loss='mse')
    # print(autoencoder.summary())
    # # fit
    # print("fit")
    # autoencoder.fit(X_train, validation_data=X_test, epochs=15, verbose=2)
    # # predict
    # print("predict")
    # test = load_lfw_dataset("data/test/")
    # predicted = predict(test[0], encoder, decoder)
    # # write
    # print("write")
    # write("data/out/first.jpg", predicted)
