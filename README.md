# pix2code-responsive
*Generating responsive websites from screenshots*
*(Originated from Tony Beltramellis pix2code)*

[![License](http://img.shields.io/badge/license-APACHE2-blue.svg)](LICENSE.txt)

## Disclaimer

The following software is shared for educational purposes only. The author and its affiliated institution are not responsible in any manner whatsoever for any damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of the use or inability to use this software.

The project pix2code is a research project demonstrating an application of deep neural networks to generate code from visual inputs.
The current implementation is not, in any way, intended, nor able to generate code in a real-world context.
We could not emphasize enough that this project is experimental and shared for educational purposes only.
Both the source code and the datasets are provided to foster future research in machine intelligence and are not designed for end users.

## Setup
### Prerequisites

- Python 3.6
- pip3

### Install dependencies

```sh
pip3 install -r requirements.txt
```

## Usage

Prepare the data:
```sh
# generate data-set according to ../data-generation/README.md

# Create directory ./datasets/responsive_web/all_data if it doesn't exist
# Copy and Paste the generated Image and gui files directly into ./datasets/responsive_web/all_data

cd /model

# split training set and evaluation set while ensuring no training example in the evaluation set
# usage: build_datasets.py <input path> <distribution (default: 6)>
./build_datasets.py ../datasets/responsive_web/

# transform images (normalized pixel values and resized pictures) in training dataset to numpy arrays (smaller files if you need to upload the set to train your model in the cloud)
# usage: convert_imgs_to_arrays.py <input path> <output path>
./convert_imgs_to_arrays.py ../datasets/responsive_web/training_set ../datasets/responsive_web/training_features
```

### Train the model
The directory /bin contains trained weights and meta data for training/sampling
If you have pretrained weights they can be placed here. The Configurations of these weights need to the same as in the model/classes/model/Config.py (e.g. Image Size)
```sh
mkdir bin

cd model

# provide input path to training data and output path to save trained model and metadata
# usage: train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>
./train.py ../datasets/responsive_web/training_set ../bin

# train on images pre-processed as arrays
./train.py ../datasets/responsive_web/training_features ../bin
```
Generate code for batch of GUIs:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./generate.py ../bin pix2code ../gui_screenshots ../code

# equivalent to command above
./generate.py ../bin pix2code ../gui_screenshots ../code greedy

# generate DSL code with beam search and a beam width of size 3
./generate.py ../bin pix2code ../gui_screenshots ../code 3
```

Generate code for a single GUI image-pair:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./sample.py ../bin pix2code ../test_gui.png ../code
```

Complete unfinished code sequence for a single GUI image:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: sample.py <trained weights path> <trained model name> <input image> <output path>
./complete_sequence.py ../bin pix2code ../test_gui_tablet.png ../test_gui_desktop.png ../code
```
### Evaluate model with evaluation set
Occasionally it might happen, that the evaluation set uses a smaller vocabulary than the trained model. In that case a new evaluation set with the same vocabulary size needs to be generated
```
# Convert evaluation set into numpy arrays
./convert_imgs_to_arrays.py ../datasets/responsive_web/eval_set ../datasets/responsive_web/eval_features

# usage: evaluate.py <input path> <trained weights>
./evaluate.py ../datasets/responsive_web/eval_features ../bin/pix2code.h5
```

Compile generated code to target language:
```sh
cd compiler

# compile .gui file to HTML/CSS (Bootstrap style)
./responsive-web-compiler.py <input file path>.gui
```
