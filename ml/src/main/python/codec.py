import os

import tensorflow as tf

DIR = "data/in/"
RESIZE_FACTOR = 1  # 4
DIM_X = int(1920 / RESIZE_FACTOR)
DIM_Y = int(1080 / RESIZE_FACTOR)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def encode():
    record_file = 'data/images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        i = 1
        for file_name in os.listdir(DIR):
            if i % 100 == 0:
                print(i)
            i += 1
            # read
            img_str = tf.io.read_file(os.path.join(DIR, file_name))
            # encode
            msg = _encode(img_str)
            # write
            writer.write(msg.SerializeToString())

def _encode(img_str):
    feature = {
        'img_str': _bytes_feature(img_str)
    }
    msg = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return msg


if __name__ == "__main__":
    encode()
