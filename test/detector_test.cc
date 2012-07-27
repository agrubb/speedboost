//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <cmath>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <ImageMagick/Magick++.h>

#include "common.h"
#include "detector.h"
#include "image_util.h"
#include "patch.h"

using namespace std;
using namespace speedboost;

const string kBoostClassifier = "/face.boost.classifier";
const string kCascadeClassifier = "/face.cascade.classifier";
const string kAnytimeClassifier = "/face.anytime.classifier";

const string kBoostMultiClassifier = "/face.boost.multi.classifier";

const string kFrame = "/seinfeld.png";

TEST(SequencerTest, SequencerTest) {
  Classifier c;
  c.ReadFromFile(FLAGS_test_data_directory + kAnytimeClassifier);

  Sequencer seq(&c);

  EXPECT_EQ(-1, seq.NextChain(0, 0.0));
  EXPECT_EQ(-1, seq.NextChain(0, 0.5));

  EXPECT_EQ(-1, seq.NextChain(1, 0.0));
  EXPECT_EQ(-1, seq.NextChain(1, 3.0));
  
  EXPECT_EQ(2, seq.NextChain(2, 0.5));
  EXPECT_EQ(3, seq.NextChain(2, 1.0));
  EXPECT_EQ(-1, seq.NextChain(2, 5.0));

  EXPECT_EQ(4, seq.NextChain(4, 0.5));
  EXPECT_EQ(-1, seq.NextChain(4, 1.0));

  EXPECT_EQ(9, seq.NextChain(9, 0.5));
  EXPECT_EQ(10, seq.NextChain(9, 1.0));
  EXPECT_EQ(-1, seq.NextChain(9, 2.0));

  EXPECT_EQ(11, seq.NextChain(11, 0.5));
  EXPECT_EQ(12, seq.NextChain(11, 0.9));
  EXPECT_EQ(13, seq.NextChain(11, 0.95));
  EXPECT_EQ(16, seq.NextChain(11, 1.1));
  EXPECT_EQ(17, seq.NextChain(11, 1.3));
  EXPECT_EQ(-1, seq.NextChain(11, 2.0));
}

// TEST(SingleScaleDetectorTest, MatchAllPatches) {
//   // Test data was trained using these values.
//   FLAGS_patch_width = 19;
//   FLAGS_patch_height = 19;
//   FLAGS_patch_depth = 1;


void OutputActivation(const Patch& activations, string filename) {
  Patch p(activations);

  for (int h = 0; h < p.height(); h++) {
    for (int w = 0; w < p.width(); w++) {
      float v = p.Value(w,h,0);
      //p.SetValue(w,h,0,exp(v) / (1.0 + exp(v)));
      p.SetValue(w, h, 0, v < 2.0 ? 0.0 : 1.0);
    }
  }

  p.WritePGM(filename);
}

void VerifyActivations(const Patch& activations, const Patch& frame,
                       const Classifier& c, float tolerance) {
  // Generate all the patches.
  vector<Patch> patches;
  vector<Label> labels;
  frame.GenerateAllPatches(FLAGS_patch_width, FLAGS_patch_height, FLAGS_patch_depth,
                           &labels, &patches);

  // Maybe use L1 or L2 error instead of straight equality.
  int incorrect = 0;
  for (int i = 0; i < (int)(patches.size()); i++) {
    float act = activations.Value(labels[i].x(), labels[i].y(), 0);
    if (c.Activation(patches[i]) != act)
      incorrect++;
  }

  // Some will be incorrect due to float rounding errors.
  EXPECT_LE(incorrect, tolerance * (float)(patches.size())) << "Too many errors in activation image";
}

TEST(DetectorTest, ComputeActivationPyramidMultiScale) {
  // Test data was trained using these values.
  FLAGS_patch_width = 19;
  FLAGS_patch_height = 19;
  FLAGS_patch_depth = 1;

  // Load the test frame and classifier.
  Magick::Image img(FLAGS_test_data_directory + kFrame);
  img.type(Magick::GrayscaleType);
  
  Classifier c;
  c.ReadFromFile(FLAGS_test_data_directory + kBoostClassifier);
  
  Patch frame(0, img.columns(), img.rows(), 1);
  ImageToPatch(img, &frame);

  Detector detect(&c, 1.0, 5, 1.3, 0.0);
  vector<Patch> activation_pyramid;

  detect.ComputeActivationPyramid(frame, &activation_pyramid);
  
  // Output images for manual inspection.
  for (int i = 0; i < (int)(activation_pyramid.size()); i++) {
    stringstream ss;
    ss << FLAGS_test_output_directory << "/detector_test.activation." << i << ".pgm";
    string filename = ss.str();
    OutputActivation(activation_pyramid[i], filename);
  }

  float current_scale = 1.0;
  for (int i = 0; i < (int)(activation_pyramid.size()); i++) {
    Patch rescaled(0, frame.width()*current_scale, frame.height()*current_scale, 1);
    Label l(0, 0, frame.width(), frame.height());

    frame.ExtractLabel(l, &rescaled);

    ASSERT_EQ(activation_pyramid[i].width(), rescaled.width());
    ASSERT_EQ(activation_pyramid[i].height(), rescaled.height());

    VerifyActivations(activation_pyramid[i], rescaled, c, 0.02);
    current_scale = current_scale / 1.3;
  }
}

TEST(DetectorTest, ComputeActivationPyramidMultiScaleMultiChannel) {
  // Test data was trained using these values.
  FLAGS_patch_width = 19;
  FLAGS_patch_height = 19;
  FLAGS_patch_depth = 3;

  // Load the test frame and classifier.
  Magick::Image img(FLAGS_test_data_directory + kFrame);
  img.type(Magick::TrueColorType);
  
  Classifier c;
  c.ReadFromFile(FLAGS_test_data_directory + kBoostMultiClassifier);
  
  Patch frame(0, img.columns(), img.rows(), 3);
  ImageToPatch(img, &frame);

  Detector detect(&c, 1.0, 5, 1.3, 0.0);
  vector<Patch> activation_pyramid;

  detect.ComputeActivationPyramid(frame, &activation_pyramid);
  
  // Output images for manual inspection.
  for (int i = 0; i < (int)(activation_pyramid.size()); i++) {
    stringstream ss;
    ss << FLAGS_test_output_directory << "/detector_test_multi.activation." << i << ".pgm";
    string filename = ss.str();
    OutputActivation(activation_pyramid[i], filename);
  }

  float current_scale = 1.0;
  for (int i = 0; i < (int)(activation_pyramid.size()); i++) {
    Patch rescaled(0, frame.width()*current_scale, frame.height()*current_scale, 3);
    Label l(0, 0, frame.width(), frame.height());

    frame.ExtractLabel(l, &rescaled);

    ASSERT_EQ(activation_pyramid[i].width(), rescaled.width());
    ASSERT_EQ(activation_pyramid[i].height(), rescaled.height());

    VerifyActivations(activation_pyramid[i], rescaled, c, 0.02);
    current_scale = current_scale / 1.3;
  }
}

TEST(DetectorTest, ComputeActivationPyramidSingleScale) {
  // Test data was trained using these values.
  FLAGS_patch_width = 19;
  FLAGS_patch_height = 19;
  FLAGS_patch_depth = 1;

  // Load the test frame and classifier.
  Magick::Image img(FLAGS_test_data_directory + kFrame);
  img.type(Magick::GrayscaleType);
  
  Classifier c;
  c.ReadFromFile(FLAGS_test_data_directory + kBoostClassifier);
  
  Patch frame(0, img.columns(), img.rows(), 1);
  ImageToPatch(img, &frame);

  Detector detect(&c, 1.0, 1, 1.0, 0.0);
  vector<Patch> activation_pyramid;
  detect.ComputeActivationPyramid(frame, &activation_pyramid);

  ASSERT_EQ(activation_pyramid.size(), 1);
  VerifyActivations(activation_pyramid[0], frame, c, 0.02);
}


TEST(DetectorTest, ComputeActivationPyramidSingleScaleMultiChannel) {
  // Test data was trained using these values.
  FLAGS_patch_width = 19;
  FLAGS_patch_height = 19;
  FLAGS_patch_depth = 3;

  // Load the test frame and classifier.
  Magick::Image img(FLAGS_test_data_directory + kFrame);
  img.type(Magick::TrueColorType);
  
  Classifier c;
  c.ReadFromFile(FLAGS_test_data_directory + kBoostMultiClassifier);
  
  Patch frame(0, img.columns(), img.rows(), 3);
  ImageToPatch(img, &frame);

  Detector detect(&c, 1.0, 1, 1.0, 0.0);
  vector<Patch> activation_pyramid;
  detect.ComputeActivationPyramid(frame, &activation_pyramid);

  ASSERT_EQ(activation_pyramid.size(), 1);
  VerifyActivations(activation_pyramid[0], frame, c, 0.02);
}
