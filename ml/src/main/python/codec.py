import os

import tensorflow as tf

DIR = "02"
DIR_X = "data/png/{0}/x/".format(DIR)
DIR_Y = "data/png/{0}/y/".format(DIR)


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
    record_file = 'data/in/{0}/images.tfrecords'.format(DIR)
    with tf.io.TFRecordWriter(record_file) as writer:
        i = 1
        for file_name_x in os.listdir(DIR_X):
            for file_name_y in os.listdir(DIR_Y):
                if i % 100 == 0:
                    print(i)
                i += 1
                # read
                img_str_x = tf.io.read_file(os.path.join(DIR_X, file_name_x))
                img_str_y = tf.io.read_file(os.path.join(DIR_Y, file_name_y))
                # encode
                msg = _encode(img_str_x, img_str_y)
                # write
                writer.write(msg.SerializeToString())


def _encode(_img_str_x, _img_str_y):
    feature = {
        'img_str_x': _bytes_feature(_img_str_x),
        'img_str_y': _bytes_feature(_img_str_y)
    }
    msg = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return msg


if __name__ == "__main__":
    encode()
