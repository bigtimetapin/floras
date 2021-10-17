import tarfile

import cv2
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
import numpy as np
from sklearn.model_selection import train_test_split

# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
IMAGES_NAME = "data/lfw-deepfunneled.tgz"

# http://vis-www.cs.umass.edu/lfw/lfw.tgz
RAW_IMAGES_NAME = "data/lfw.tgz"


def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):
    # Read photos
    all_photos = []

    # tqdm in used to show progress bar while reading the data in a notebook here, you can change
    # tqdm_notebook to use it outside a notebook
    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in f.getmembers():
            # Only process image files from the compressed data
            if m.isfile() and m.name.endswith(".jpg"):
                print(m.name)
                # Prepare image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())

                # Crop only faces and resize it
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))

                all_photos.append(img)

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
    cv2.imwrite(file_name, img)


if __name__ == "__main__":
    # read photos
    X = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)
    # normalize
    X = X.astype('float32') / 255.0 - 0.5
    # split train,test
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
    # get dimensions
    IMG_SHAPE = X.shape[1:]
    # build auto encoder
    encoder, decoder = build_autoencoder(IMG_SHAPE, 32)
    inp = Input(IMG_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)
    autoencoder = Model(inp, reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')
    print(autoencoder.summary())
    # fit
    autoencoder.fit(x=X_train, y=X_train, epochs=3, validation_data=(X_test, X_test))
    # predict
    predicted = predict(X_test[0], encoder, decoder)
    # write
    write("data/out/first.jpg", predicted)
