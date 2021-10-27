import os

import cv2
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
import numpy as np
from sklearn.model_selection import train_test_split

# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
DIR = "data/in/"
DIM_X = 1920
DIM_Y = 1080
COMPRESSION_FACTOR = 32

def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_lfw_dataset():
    i = 1
    all_photos = []
    for file_name in os.listdir(DIR):
        if i % 100 == 0:
            print(i)
        i += 1
        # read photo
        img = cv2.imread(os.path.join(DIR, file_name))
        # Prepare image
        # Crop only faces and resize it
        # img = img[dy:-dy, dx:-dx]
        img = cv2.resize(img, (DIM_X, DIM_Y))
        # append
        all_photos.append(img)
    # encode as stack uint8
    all_photos = np.stack(all_photos).astype('uint8')
    return all_photos

"""
TODO: try this stuff
stacked_ae = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
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
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))
    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape)))
    decoder.add(Reshape(img_shape))
    return encoder, decoder


def predict(img, encoder, decoder):
    code = encoder.predict(img[None])[0]
    return decoder.predict(code[None])[0]


def write(file_name, img):
    img = (img + 0.5) * 255.0
    cv2.imwrite(file_name, img)


if __name__ == "__main__":
    # read photos
    print("reading data . . ")
    X = load_lfw_dataset()
    # normalize
    print("normalizing data . . ")
    X = X.astype('float32') / 255.0 - 0.5
    # split train,test
    print("split train, test")
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
    # get dimensions
    print("get dimensions")
    IMG_SHAPE = X.shape[1:]
    # build auto encoder
    print("build auto encoder")
    encoder, decoder = build_autoencoder(IMG_SHAPE, COMPRESSION_FACTOR)
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
    autoencoder.fit(x=X_train, y=X_train, epochs=3, validation_data=(X_test, X_test))
    # predict
    print("predict")
    predicted = predict(X_test[1], encoder, decoder)
    # write
    print("write")
    write("data/out/first.jpg", predicted)
