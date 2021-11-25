import tensorflow as tf


RESIZE_FACTOR = 4  # 4
DIM_X = int(1920 / RESIZE_FACTOR)
DIM_Y = int(1080 / RESIZE_FACTOR)


def _decode(img_str):
    img = tf.image.decode_jpeg(img_str)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [DIM_X, DIM_Y])
    return img
