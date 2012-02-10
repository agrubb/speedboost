//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_FEATURE_H
#define SPEEDBOOST_FEATURE_H

#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "feature.pb.h"
#include "patch.h"

namespace speedboost {

/**
 * Single rectangle for Haar based features.
 * Upper left corner at (x0, y0),
 * lower right at (x1, y1).
 */
class Box {
public:
  Box() {}
  
  Box(int x0, int y0, int x1, int y1)
    : x0_(x0), y0_(y0), x1_(x1), y1_(y1) {
    assert(x0 >= 0 && x0 < FLAGS_patch_width);
    assert(x1 >= 0 && x1 < FLAGS_patch_width);
    assert(y0 >= 0 && y0 < FLAGS_patch_height);
    assert(y1 >= 0 && y1 < FLAGS_patch_height);
  }
  
  int x0_, y0_, x1_, y1_;

  /**
   * Convert to and from protobuf representation.
   */
  bool FromMessage(const BoxMessage& msg);
  void ToMessage(BoxMessage* msg) const;
  
  /**
   * File input and output.
   */
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;
};


/**
 * Haar wavelet based feature.
 * The feature will evaluate to: w0 * area(b0) + w1 * area(b1),
 * using channel c of the patch (for multi-channel images).
 */
class Feature {
public:
  Feature() {}

  Feature(const Box& b0, const Box& b1, float w0, float w1, int c)
    : b0_(b0), b1_(b1), w0_(w0), w1_(w1), c_(c) {
    assert(c >= 0 && c < FLAGS_patch_depth);
  }

  void Print() const {
    printf("%f*integral[(%d, %d) -> (%d, %d)] + ",
           w0_, b0_.x0_, b0_.y0_, b0_.x1_, b0_.y1_);
    printf("%f*integral[(%d, %d) -> (%d, %d)]",
           w1_, b1_.x0_, b1_.y0_, b1_.x1_, b1_.y1_);
    printf(" (chan %d)", c_);
  }

  /**
   * Evaluate this feature on patch p.
   * Assumes that p is an extracted patch suitable for
   * training, etc.
   */
  float Evaluate(const Patch& p) const;

  /**
   * Convert to and from protobuf representation.
   */
  bool FromMessage(const FeatureMessage& msg);
  void ToMessage(FeatureMessage* msg) const;

  /**
   * File input and output.
   */  
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;

  /**
   * Read or write a vector of features to a file.
   */
  static int ReadFeaturesFromFile(const std::string& filename, std::vector<Feature>* features);
  static void WriteFeaturesToFile(const std::string& filename, const std::vector<Feature>& features);

  /**
   * Generate num_features features and put them in the provided vector.
   * Currently this generates two random rectangles
   */
  static void GenerateFeatures(int num_features, std::vector<Feature>* features);
  
  Box b0_, b1_;
  float w0_, w1_;
  int c_;
};

}  // namespace speedboost

#endif  // ifndef SPEEDBOOST_FEATURE_H
