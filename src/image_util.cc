//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <cassert>
#include <iostream>
#include <gflags/gflags.h>
#include <ImageMagick/Magick++.h>

#include "image_util.h"
#include "patch.h"

namespace speedboost {

void ImageToPatch(const Magick::Image& image, Patch* patch) {
  assert(patch->width() == (int)(image.columns()));
  assert(patch->height() == (int)(image.rows()));
  if (image.type() == Magick::TrueColorType) {
    assert(patch->channels() == 3);
    const Magick::PixelPacket* pixel = image.getConstPixels(0, 0, image.columns(), image.rows());
    for (int h = 0; h < (int)(image.rows()); h++) {
      for (int w = 0; w < (int)(image.columns()); w++, pixel++) {
	patch->SetValue(w, h, 0, Magick::Color::scaleQuantumToDouble(pixel->red));
	patch->SetValue(w, h, 1, Magick::Color::scaleQuantumToDouble(pixel->green));
	patch->SetValue(w, h, 2, Magick::Color::scaleQuantumToDouble(pixel->blue));
      }
    }
  } else if (image.type() == Magick::GrayscaleType) {
    assert(patch->channels() == 1);
    const Magick::PixelPacket* pixel = image.getConstPixels(0, 0, image.columns(), image.rows());
    for (int h = 0; h < (int)(image.rows()); h++) {
      for (int w = 0; w < (int)(image.columns()); w++, pixel++) {
	patch->SetValue(w, h, 0, Magick::Color::scaleQuantumToDouble(pixel->red));
      }
    }
  }
}

void PatchToImage(const Patch& patch, Magick::Image* image) {
  assert(patch.width() == (int)(image->columns()));
  assert(patch.height() == (int)(image->rows()));
  if (image->type() == Magick::TrueColorType) {
    assert(patch.channels() == 3);
    Magick::PixelPacket* pixel = image->getPixels(0, 0, patch.width(), patch.height());
    for (int h = 0; h < patch.height(); h++) {
      for (int w = 0; w < patch.width(); w++, pixel++) {
	pixel->red   = Magick::Color::scaleDoubleToQuantum(patch.Value(w,h,0));
	pixel->green = Magick::Color::scaleDoubleToQuantum(patch.Value(w,h,1));
	pixel->blue  = Magick::Color::scaleDoubleToQuantum(patch.Value(w,h,2));
      }
    }
  } else if (image->type() == Magick::GrayscaleType) {
    assert(patch.channels() == 1);
    Magick::PixelPacket* pixel = image->getPixels(0, 0, patch.width(), patch.height());
    for (int h = 0; h < patch.height(); h++) {
      for (int w = 0; w < patch.width(); w++, pixel++) {
	pixel->red   = Magick::Color::scaleDoubleToQuantum(patch.Value(w,h,0));
	pixel->green = Magick::Color::scaleDoubleToQuantum(patch.Value(w,h,0));
	pixel->blue  = Magick::Color::scaleDoubleToQuantum(patch.Value(w,h,0));
      }
    }
  }
}

}  //namespace speedboost
