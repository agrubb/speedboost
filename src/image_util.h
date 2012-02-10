//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_IMAGE_UTIL_H
#define SPEEDBOOST_IMAGE_UTIL_H

#include <gflags/gflags.h>
#include <ImageMagick/Magick++.h>

#include "patch.h"

namespace speedboost {

void ImageToPatch(const Magick::Image& image, Patch* patch);
void PatchToImage(const Patch& patch, Magick::Image* image);

}  //namespace speedboost

#endif  // #ifndef SPEEDBOOST_IMAGE_UTIL_H
