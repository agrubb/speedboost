Prerequisites
-------------

You need the following packages, or their equivalents, installed:

- Magick++ C++ Library -- libmagick++-dev on Ubuntu
- Google Protobuf Compiler + Library -- protobuf-compiler and libprotobuf-dev on Ubuntu
- Google Command Line Flags -- http://code.google.com/p/gflags/, definitely works with versions after 1.6


Installation
------------

You can build the code by running 'make' in the root directory.
The first time this may take a while, as building the thirdparty code
can take some time.  You can use 'make test' to run the (minimal)
test code and make sure everything is working.

Binaries
--------

Building the code will generate a number of binaries in the bin/ directory:

- check -- Runs the automated tests.
- load -- Takes raw images and labels and stores them in a binary format
  for training and testing.
- train -- Takes patches load using load and trains a predictor on them.
- predict -- Takes loaded patches and a classifier, and generates statistics
  on prediction performance, etc.
- detect -- Takes an image and a classifier and runs actual object detection
  on the image.  Can output either a heatmap or labeled detections (not working yet).

