#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from classes.dataset.Dataset import Dataset
from classes.Vocabulary import Vocabulary

__author__ = 'Kevin Kössl'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys

from classes.model.pix2code import *
from classes.dataset.Dataset import *

def run(input_path, trained_model):

    dataset = Dataset()

    dataset.load(input_path, generate_binary_sequences=True)
    dataset.convert_arrays()

    input_shape = dataset.input_shape
    output_size = dataset.output_size

    print(len(dataset.input_images_tablet), len(dataset.input_images_desktop), len(dataset.partial_sequences), len(dataset.next_words))
    print(dataset.input_images_tablet.shape, dataset.input_images_desktop.shape, dataset.partial_sequences.shape, dataset.next_words.shape)


    model = pix2code(input_shape, output_size, "")

    if trained_model is not None:
        model.model.load_weights(trained_model)

    evaluation = model.evaluate(dataset.input_images_tablet, dataset.input_images_desktop, dataset.partial_sequences, dataset.next_words)

    correct_samples = 0
    for i in range(0, len(dataset.input_images_tablet)):
        print("Predicting {}".format(i))
        probas = model.predict(np.array([dataset.input_images_tablet[i]]), np.array([dataset.input_images_desktop[i]]), np.array([dataset.partial_sequences[i]]))

        prediction = np.argmax(probas)

        sparse_label = np.zeros(output_size)
        sparse_label[prediction] = 1

        if np.array_equal(sparse_label, dataset.next_words[i]):
            correct_samples += 1

    print("loss: {}".format(evaluation))
    accuracy = correct_samples / len(dataset.partial_sequences)
    print("accuracy: {}".format(accuracy))


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("evaluate.py <input path> <trained weights (optional)>")
        exit(0)
    else:
        input_path = argv[0]
        trained_weigths = argv[1]


    run(input_path, trained_weigths)
