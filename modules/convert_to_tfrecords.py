
import os
import random
import string

import numpy as np
import cv2
import tensorflow as tf

EXAMPLE = "MNIST-data/training/0/1.png"
INPUT_PATH = "claptcha/test/"
OUTPUT_PATH = "claptcha/records/"
OUTPUT_NAME = "test.tfrecord"


def indexletters(letters: str) -> list:
    return [string.ascii_uppercase.index(c.upper()) for c in letters]


def read_filenames(path: str) -> list:
    """Reads the file names of the images located in a folder.
    Args:
      path: Location of the folder that contains the images
    Returns:
      List of filenames with path
      List of labels
    """
    return [{"filename": os.path.join(root, name),
             "label": indexletters(name[:-4])}
            for root, dirs, files in os.walk(path)
            for name in files if os.path.isfile(os.path.join(root, name))]


def load_file(filename: str) -> np.ndarray:
    img = cv2.imread(filename=filename)
    assert isinstance(img, np.ndarray)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (80, 28), interpolation=cv2.INTER_AREA)
    return img


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def bytes_list_feature(list_):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_))


def int64_list_feature(list_):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_))


def float_list_feature(list_):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_))


def write_tfrecord(*, path: str, data: list):

    with tf.python_io.TFRecordWriter(path=path) as writer:

        for i in range(len(data)):
            if not i % 1000:
                print("Data written: {0}/{1}".format(i, len(data)))

            img = load_file(data[i]["filename"]).tobytes()
            label = data[i]["label"]

            feature = {
                'label': int64_list_feature(label),
                'image': bytes_feature(tf.compat.as_bytes(img))
            }

            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
        print("Finished")


files = read_filenames(INPUT_PATH)
random.shuffle(files)
write_tfrecord(path=OUTPUT_PATH+OUTPUT_NAME, data=files)
