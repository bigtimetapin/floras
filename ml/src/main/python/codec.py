import os

import tensorflow as tf

JUST_X = True

DIRS = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20"
]


def dir_x(n):
    return "data/png/{0}/x".format(n)


def dir_y(n):
    return "data/png/{0}/y".format(n)


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
    record_file = 'data/in/images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        i = 1
        for n in DIRS:
            if not JUST_X:
                _dir_x = dir_x(n)
                _dir_y = dir_y(n)
                for file_name_x in os.listdir(_dir_x):
                    for file_name_y in os.listdir(_dir_y):
                        if i % 100 == 0:
                            print(i)
                            print("y")
                        i += 1
                        # read
                        img_str_x = tf.io.read_file(os.path.join(_dir_x, file_name_x))
                        img_str_y = tf.io.read_file(os.path.join(_dir_y, file_name_y))
                        # encode
                        msg = _encode(img_str_x, img_str_y)
                        # write
                        writer.write(msg.SerializeToString())

            else:
                _dir_x = dir_x(n)
                for file_name_x in os.listdir(_dir_x):
                    if i % 100 == 0:
                        print(i)
                        print(_dir_x)
                    i += 1
                    # read
                    img_str_x = tf.io.read_file(os.path.join(_dir_x, file_name_x))
                    # encode
                    msg = _encode(img_str_x, img_str_x)
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
