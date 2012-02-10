//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <cmath>
#include <gflags/gflags.h>
#include <ImageMagick/Magick++.h>
#include <iostream>
#include <string>

#include "classifier.h"
#include "detector.h"
#include "feature.h"
#include "image_util.h"
#include "patch.h"

using namespace speedboost;
using namespace std;

DEFINE_string(frame_filename, "",
              "Image file containing the test frame.");
DEFINE_string(classifier_filename, "",
              "File containing the trained classifier.");
DEFINE_string(activations_filename, "activations.pgm",
	      "File to output the merged activation image too.");
DEFINE_int32(num_scales, 3,
             "Number of scales in image pyramid.");
DEFINE_double(scaling_factor, 1.2,
              "Factor that each image scales down by in pyramid, "
              "i.e. successive levels are scaling_factor apart in size.");

int main(int argc, char*argv[])
{
  // parse up the flags
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Load the test frame and classifier.
  Magick::Image img(FLAGS_frame_filename);
  if (FLAGS_patch_depth == 3) {
    img.type(Magick::TrueColorType);
  } else {
    img.type(Magick::GrayscaleType);
  }

  Patch frame(0, img.columns(), img.rows(), FLAGS_patch_depth);
  ImageToPatch(img, &frame);

  Classifier c;
  c.ReadFromFile(FLAGS_classifier_filename);

  Detector detector(&c);
  Patch activations(0, frame.width(), frame.height(), 1);
  detector.ComputeMergedActivation(frame, FLAGS_num_scales, FLAGS_scaling_factor, &activations);

  // Transform the activations with a sigmoid for outputs in [0,1].
  for (int w = 0; w < activations.width(); w++) {
    for (int h = 0; h < activations.height(); h++) {
      float a = activations.Value(w, h, 0);
      activations.SetValue(w, h, 0, exp(a) / (1.0 + exp(a)));
    }
  }
  activations.WritePGM(FLAGS_activations_filename);

  return 0;
}
