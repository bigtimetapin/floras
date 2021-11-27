import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
import numpy as np
from preprocess import _decode, DIM_X, DIM_Y

_DIR = "02"
DIR = "data/in/{0}".format(_DIR)
COMPRESSION_FACTOR = int(32 * 5)  # * 5)
IMG_SHAPE = tf.TensorShape((DIM_X, DIM_Y, 3))
VALIDATION_SIZE = 300

image_feature_description = {
    'img_str_x': tf.io.FixedLenFeature([], tf.string),
    'img_str_y': tf.io.FixedLenFeature([], tf.string)
}


def decode(proto):
    img = tf.py_function(
        _decode,
        [proto],
        tf.float32
    )
    return img

def _dataset():
    # encode as dataset
    __dataset = tf.data.TFRecordDataset('{0}/images.tfrecords'.format(DIR))
    # parse
    __dataset = __dataset.map(lambda x: tf.io.parse_single_example(x, image_feature_description))
    # decode
    __dataset = __dataset.map(lambda x: (decode(x['img_str_x']), decode(x['img_str_y'])))
    return __dataset

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


def build_autoencoder(code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(input_shape=IMG_SHAPE))
    encoder.add(Flatten())
    encoder.add(Dense(code_size, ))
    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer(input_shape=(code_size,)))
    decoder.add(Dense(np.prod(IMG_SHAPE)))
    decoder.add(Reshape(IMG_SHAPE))
    return encoder, decoder


if __name__ == "__main__":
    tf.autograph.set_verbosity(10)
    # read photos
    print("reading data . . ")
    dataset = _dataset()
    for row in dataset.take(1):
        print(row[0])
    # split train,test
    print("split train, test")
    dataset = dataset.shuffle(buffer_size=1000, seed=19)
    X_validation = dataset.take(VALIDATION_SIZE)
    X_train = dataset.skip(VALIDATION_SIZE)
    # build auto encoder
    print("build auto encoder")
    encoder, decoder = build_autoencoder(COMPRESSION_FACTOR)
    print("inp")
    inp = Input(IMG_SHAPE)
    print("code")
    code = encoder(inp)
    print("reconstruction")
    reconstruction = decoder(code)
    print("autoencoder")
    autoencoder = Model(inp, reconstruction)
    print("compile")
    autoencoder.compile(optimizer='adamax', loss='mse')
    print(autoencoder.summary())
    # fit
    print("fit")
    autoencoder.fit(x=X_train, validation_data=X_validation, epochs=3, verbose=2)
    # save
    print("write")
    encoder.save("data/{0}/model/encoder".format(_DIR))
    decoder.save("data/{0}/model/decoder".format(_DIR))
