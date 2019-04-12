#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules.input_function import my_input_fn
from modules.model import cnn_model_fn

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(unused_argv):
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="mnist_convnet_model"
    )

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    def train_input_fn():
        return my_input_fn(
            filenames=["claptcha/records/train.tfrecord"],
            batch_size=100,
            shuffle=True,
            num_epochs=None)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    def eval_input_fn():
        return my_input_fn(
            filenames=["claptcha/records/train.tfrecord"],
            shuffle=False,
            num_epochs=1)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # predict_input_fn = lambda: my_input_fn(
    #    filenames = ["records/test.tfrecord"],
    #    shuffle=False,
    #    num_epochs=1)

    #predictions  = mnist_classifier.predict(input_fn=predict_input_fn)
    # for p in predictions:
    #    print(labels_text(p["classes"], p["labels"]))


if __name__ == "__main__":
    tf.app.run()
