//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_PATCH_H
#define SPEEDBOOST_PATCH_H

#include <cassert>
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "patch.pb.h"

DECLARE_int32(patch_width);
DECLARE_int32(patch_height);
DECLARE_int32(patch_depth);

namespace speedboost {

/**
 * Labeled rectangle in a patch,
 * starting at (x,y) with width w and height h.
 */
class Label {
public:
  Label() {}

  Label(int x, int y, int w, int h, char label = 0)
    : x_(x), y_(y), w_(w), h_(h), label_(label) {
  }

  bool operator==(const Label& other) const {
    return (x_ == other.x_ &&
            y_ == other.y_ &&
            w_ == other.w_ &&
            h_ == other.h_);
  }

  /**
   * Convert to and from protobuf representation.
   */
  bool FromMessage(const LabelMessage& msg);
  void ToMessage(LabelMessage* msg) const;

  /**
   * File input and output.
   */
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;

  inline int x() const { return x_; }
  inline int y() const { return y_; }
  inline int w() const { return w_; }
  inline int h() const { return h_; }
  inline char label() const { return label_; }

private:
  int x_, y_, w_, h_;
  char label_;
};


/**
 * An image patch or entire image frame.
 * Has width * height * channels pixels.
 */
class Patch {
public:
  Patch()
    : label_(0), width_(0), height_(0), channels_(0),
      data_() {
  }

  Patch(char label, int w, int h, int c)
    : label_(label), width_(w), height_(h), channels_(c),
      data_(w*h*c, 0) {
  }

  Patch(const Patch& other) {
    label_ = other.label_;

    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;

    data_ = other.data_;
  }
  
  bool operator==(const Patch& other) const {
    return false;
  }

  Patch& operator=(const Patch& other) {
    label_ = other.label_;

    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;

    data_ = other.data_;
    return *this;
  }

  inline void SetValue(int w, int h, int c, float v) {
    assert((c * width_ * height_ + h * width_ + w) < (int)(data_.size()));
    assert((c * width_ * height_ + h * width_ + w) >= 0);
    data_[c * width_ * height_ + h * width_ + w] = v;
  }

  inline float Value(int w, int h, int c) const {
    return data_[c * width_ * height_ + h * width_ + w];
  }

  /**
   * Compute the integal image from the data stored in this patch.
   */
  void ComputeIntegralImage();

  /**
   * Extract the rectangle given in label and store the data
   * in patch.  If the size of patch and label are different,
   * extract label will scale the extracted area up/down as
   * appropriate.
   */
  void ExtractLabel(const Label& label, Patch* patch) const;

  /**
   * Extract every patch of size [width x height], with step pixels between
   * each adjacent patch.
   */
  void GenerateAllPatches(int width, int height, int step, std::vector<Label>* labels,
                          std::vector<Patch>* patches) const;

  /**
   * Convert to and from protobuf representation.
   */
  bool FromMessage(const PatchMessage& msg);
  void ToMessage(PatchMessage* msg) const;

  /**
   * File input and output.
   */
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;

  /**
   * Write patch out as raw pgm/ppm image.
   */
  bool WritePPM(std::string filename) const;
  bool WritePGM(std::string filename) const;

  inline void set_label(char label) { label_ = label; }

  inline char label() const { return label_; }
  inline int width() const { return width_; }
  inline int height() const { return height_; }
  inline int channels() const { return channels_; }

  friend class SingleScaleDetector;
  friend class Detector;
  friend class Feature;

protected:
  void ExtractLabelArea(const Label& label, Patch* patch) const;
  void ExtractLabelInterp(const Label& label, Patch* patch) const;

  char label_;
  int width_, height_, channels_;
  std::vector<float> data_;
};

}  // namespace speedboost

#endif  // ifndef SPEEDBOOST_PATCH_H
