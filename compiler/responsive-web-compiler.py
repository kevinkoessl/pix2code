#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Kevin Kössl'

import sys

from os.path import basename
from classes.Utils import *
from classes.Compiler import *

if __name__ == "__main__":
    argv = sys.argv[1:]
    length = len(argv)
    if length != 0:
        input_file = argv[0]
    else:
        print("Error: not enough argument supplied:")
        print("responsive-web-compiler.py <path> <file name>")
        exit(0)

FILL_WITH_RANDOM_TEXT = True
TEXT_PLACE_HOLDER = "[]"

dsl_path = "assets/responsive-web-dsl-mapping.json"
compiler = Compiler(dsl_path)


def render_content_with_text(key, value):
    if FILL_WITH_RANDOM_TEXT:
        if key.find("btn") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
        elif key.find("link") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(space_number=0))
        elif key.find("title") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=20, space_number=3))
        elif key.find("text") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
        elif key.find("card-header") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=23, space_number=3))
        elif key.find("card-footer") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=20, space_number=7))
    return value

file_uid = basename(input_file)[:basename(input_file).find(".")]
path = input_file[:input_file.find(file_uid)]

input_file_path = "{}{}.gui".format(path, file_uid)
output_file_path = "{}{}.html".format(path, file_uid)

compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)
