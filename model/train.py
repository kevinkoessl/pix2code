#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from classes.dataset.Dataset import Dataset
from classes.Vocabulary import Vocabulary

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'
__modified__ = 'Kevin Kössl'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import sys

from classes.model.pix2code import *
from classes.dataset.Dataset import *

#adjusted in order to deal with two input images, removed the option to train with a generator
def run(input_path, output_path, pretrained_model=None):

    dataset = Dataset()

    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)


    dataset.convert_arrays()

    input_shape = dataset.input_shape
    output_size = dataset.output_size

    print(len(dataset.input_images_tablet), len(dataset.input_images_desktop), len(dataset.partial_sequences), len(dataset.next_words))
    print(dataset.input_images_tablet.shape, dataset.input_images_desktop.shape, dataset.partial_sequences.shape, dataset.next_words.shape)


    model = pix2code(input_shape, output_size, output_path)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)

    model.fit(dataset.input_images_tablet, dataset.input_images_desktop, dataset.partial_sequences, dataset.next_words)


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        pretrained_weigths = None if len(argv) < 4 else argv[3]

    run(input_path, output_path, pretrained_model=pretrained_weigths)
