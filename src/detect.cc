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

DEFINE_bool(compute_activations, false,
            "Compute the activation image.");
DEFINE_string(activation_image_filename, "activation.pgm",
	      "File to output the merged activation image to.");

DEFINE_bool(compute_detections, true,
            "Compute the detections for input frame.");
DEFINE_string(detection_image_filename, "detections.ppm",
	      "File to output the detection image (drawn rectangles) to.");

DEFINE_bool(compute_updates, false,
            "Compute the updates (# of features computed per pixel) for input frame.");
DEFINE_string(update_image_filename, "updates.ppm",
	      "File to output the update image (stand in for work performed) to.");
DEFINE_double(max_updates, 255,
              "Value to use as the maximum number of updates any single pixel will see.");

DEFINE_double(initial_scale, 1.0,
              "The initial scale to start detection objects at.  "
              "If not changed from the default, the scale will be calculated adaptively "
              "using smallest_detection_ratio.");
DEFINE_double(smallest_detection_ratio, 0.1,
              "If initial_scale not changed from default, this ratio will be used to calculate "
              "an initial_scale that corresponds to the smallest detection having this fraction "
              "of the entire image area.");
DEFINE_int32(num_scales, 3,
             "Number of scales in image pyramid.");
DEFINE_double(scaling_factor, 1.2,
              "Factor that each image scales down by in pyramid, "
              "i.e. successive levels are scaling_factor apart in size.");
DEFINE_double(detection_threshold, 0.0,
              "Any patches with activation > detection_threshold are considered "
              "to be positive detections.");

void DrawDetection(const Label& det, Patch* image)
{
  int x1 = det.x();
  int y1 = det.y();
  int x2 = det.x() + det.w() - 1;
  int y2 = det.y() + det.h() - 1;

  // Top and bottom line
  for (int x = x1; x <= x2; x++) {
    image->SetValue(x, y1, 0, 0.5);
    image->SetValue(x, y1, 1, 1.0);
    image->SetValue(x, y1, 2, 0.5);

    image->SetValue(x, y2, 0, 0.5);
    image->SetValue(x, y2, 1, 1.0);
    image->SetValue(x, y2, 2, 0.5);
  }

  for (int y = y1; y <= y2; y++) {
    image->SetValue(x1, y, 0, 0.5);
    image->SetValue(x1, y, 1, 1.0);
    image->SetValue(x1, y, 2, 0.5);

    image->SetValue(x2, y, 0, 0.5);
    image->SetValue(x2, y, 1, 1.0);
    image->SetValue(x2, y, 2, 0.5);
  }
}

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

  if (google::GetCommandLineFlagInfoOrDie("initial_scale").is_default) {
    float smallest_area = frame.width() * frame.height() * FLAGS_smallest_detection_ratio;
    float patch_area = FLAGS_patch_width * FLAGS_patch_height;
    float scale = sqrt(smallest_area / patch_area);
    cout << "Setting FLAGS_initial_scale to: " << scale << endl;
    FLAGS_initial_scale = scale;
  }

  Detector detector(&c, FLAGS_initial_scale, FLAGS_num_scales, FLAGS_scaling_factor, FLAGS_detection_threshold);

  if (FLAGS_compute_detections) {
    vector<Label> detections;
    detector.ComputeDetections(frame, &detections);

    cout << "Detections:" << endl;
    for (int i = 0; i < (int)(detections.size()); i++) {
      cout << "(" << detections[i].x() << "," << detections[i].y() << ")"
           << " [" << detections[i].w() << "x" << detections[i].h() << "]" << endl;
    }

    if (FLAGS_detection_image_filename != "") {
      // Load the test frame and classifier.
      Magick::Image color_img(FLAGS_frame_filename);
      color_img.type(Magick::TrueColorType);

      Patch detection_image(0, img.columns(), img.rows(), 3);
      ImageToPatch(color_img, &detection_image);
        
      for (int i = 0; i < (int)(detections.size()); i++) {
        DrawDetection(detections[i], &detection_image);
      }

      detection_image.WritePPM(FLAGS_detection_image_filename);
    }
  }

  if (FLAGS_compute_activations) {
    Patch activations(0, frame.width(), frame.height(), 1);
    detector.ComputeMergedActivation(frame, &activations);
    
    // Transform the activations with a sigmoid for outputs in [0,1].
    for (int w = 0; w < activations.width(); w++) {
      for (int h = 0; h < activations.height(); h++) {
        float a = activations.Value(w, h, 0);
        activations.SetValue(w, h, 0, exp(a) / (1.0 + exp(a)));
      }
    }
    activations.WritePGM(FLAGS_activation_image_filename);
  }

  if (FLAGS_compute_updates) {
    Patch updates(0, frame.width(), frame.height(), 1);
    detector.ComputeMergedUpdates(frame, &updates);
    
    // Transform the updates by dividing by max.
    for (int w = 0; w < updates.width(); w++) {
      for (int h = 0; h < updates.height(); h++) {
        float a = updates.Value(w, h, 0);
        updates.SetValue(w, h, 0, a / FLAGS_max_updates);
      }
    }
    updates.WritePGM(FLAGS_update_image_filename);
  }

  return 0;
}
