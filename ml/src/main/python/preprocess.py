import tensorflow as tf


def _decode(img_str, _dim_x, _dim_y):
    # img_str = proto['img_str']
    img = tf.image.decode_jpeg(img_str)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [_dim_x, _dim_y])
    return img
