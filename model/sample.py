#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'
__modified__ = 'Kevin Kössl'

import sys

from os.path import basename
from classes.Sampler import *
from classes.model.pix2code import *

argv = sys.argv[1:]

if len(argv) < 4:
    print("Error: not enough argument supplied:")
    print("sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path_tablet = argv[2]
    input_path_desktop = argv[3]
    output_path = argv[4]
    search_method = "greedy" if len(argv) < 6 else argv[5]

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# restore np.load for future normal usage
np.load = np_load_old

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path))
input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = pix2code(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

file_name = basename(input_path_tablet)[:basename(input_path_tablet).find(".")]

#adjusted in order to deal with two input images
evaluation_img_tablet = Utils.get_preprocessed_img(input_path_tablet, IMAGE_SIZE)
evaluation_img_desktop = Utils.get_preprocessed_img(input_path_desktop, IMAGE_SIZE)

if search_method == "greedy":
    result, _ = sampler.predict_greedy(model, np.array([evaluation_img_tablet]), np.array([evaluation_img_desktop]))
    print("Result greedy: {}".format(result))
else:
    beam_width = int(search_method)
    print("Search with beam width: {}".format(beam_width))
    result, _ = sampler.predict_beam_search(model, np.array([evaluation_img_tablet]), np.array([evaluation_img_desktop]), beam_width=beam_width)
    print("Result beam: {}".format(result))

with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
    out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))
