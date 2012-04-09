//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "patch.h"
#include "patch.pb.h"
#include "util.h"

using namespace std;
using namespace google::protobuf;
using namespace google::protobuf::io;

DEFINE_int32(patch_width, 24,
	     "Width of extracted patches.");
DEFINE_int32(patch_height, 24,
	     "Height of extracted patches.");
DEFINE_int32(patch_depth, 1,
	     "Depth of extracted patches.");

namespace speedboost {

bool Label::FromMessage(const LabelMessage& msg) {
  x_ = msg.x();
  y_ = msg.y();
  w_ = msg.w();
  h_ = msg.h();
  label_ = msg.label();

  return true;
}

void Label::ToMessage(LabelMessage* msg) const {
  msg->set_x(x_);
  msg->set_y(y_);
  msg->set_w(w_);
  msg->set_h(h_);
  msg->set_label(label_);
}

bool Label::Read(istream& in) {
  LabelMessage msg;
  
  return ReadMessage(in, &msg) && FromMessage(msg);
}

void Label::Write(ostream& out) const {
  LabelMessage msg;
  
  ToMessage(&msg);
  WriteMessage(out, msg);
}

void Patch::ComputeIntegralImage() {
  for (int c = 0; c < channels_; c++) {
    for (int h = 0; h < height_; h++) {
      float row_total = 0;
      for (int w = 0; w < width_; w++) {
	float prev = (h > 0) ? Value(w, h-1, c) : 0.0;
	row_total += Value(w, h, c);
	SetValue(w, h, c, row_total + prev);
      }
    }
  }
}

  void Patch::ExtractLabel(const Label& l, Patch* p, bool nearest) const {
  assert(channels() == p->channels());
  
  if ((l.w() == p->width()) && (l.h() == p->height())) {
    for (int x = 0; x < p->width(); x++) {
      for (int y = 0; y < p->height(); y++) {
	for (int c = 0; c < channels(); c++) {
	  p->SetValue(x, y, c, Value(x + l.x(), y + l.y(), c));
	}
      }
    }
  } else if (nearest) {
    ExtractLabelNearest(l, p);
  } else if ((l.w() > p->width()) && (l.h() > p->height())) {
    // Label is larger than patch, i.e. we are shrinking the image.
    ExtractLabelArea(l, p);
  } else {
    // At least one dimension is getting bigger,
    // just use linear interpolation.
    ExtractLabelInterp(l, p);
  }
}

void Patch::ExtractLabelArea(const Label& l, Patch* p) const {
  int lw = l.w();
  int lh = l.h();
  int pw = p->width();
  int ph = p->height();
  int x0 = l.x();
  int y0 = l.y();
  float xscale = float(lw)/float(pw);
  float yscale = float(lh)/float(ph);

  // Squash the x dimension in to a pw x lh temporary patch.
  Patch buf(0, pw, lh, channels());
  for (int x = 0; x < pw; x++) {
    for (int y = 0; y < lh; y++) {
      for (int c = 0; c < channels(); c++) {
	buf.SetValue(x, y, c, 0.0);
      }
    }
  }

  float rem = 0.0;
  int px = 0;
  for (int x = 0; x < lw; x++) {
    if ((rem + 1) < xscale) {
      for (int y = 0; y < lh; y++) {
	for (int c = 0; c < channels(); c++) {
	  buf.SetValue(px, y, c, buf.Value(px, y, c) + Value(x + x0, y + y0, c));
	}
      }
      rem += 1;
    } else {
      float alpha = xscale - rem;
      //float alpha = rem - floor(rem);

      for (int y = 0; y < lh; y++) {
	for (int c = 0; c < channels(); c++) {
	  buf.SetValue(px, y, c, buf.Value(px, y, c) + alpha * Value(x + x0, y + y0, c));
	}
      }
      if (px < pw - 1) {
	for (int y = 0; y < lh; y++) {
	  for (int c = 0; c < channels(); c++) {
	    buf.SetValue(px + 1, y, c, (1 - alpha) * Value(x + x0, y + y0, c));
	  }
	}
      }
      px++;
      rem = 1 - alpha;
    }
  }

  // Now squash the y dimension down.
  for (int x = 0; x < pw; x++) {
    for (int y = 0; y < ph; y++) {
      for (int c = 0; c < channels(); c++) {
	p->SetValue(x, y, c, 0.0);
      }
    }
  }

  rem = 0.0;
  int py = 0;
  for (int y = 0; y < lh; y++) {
    if ((rem + 1) < yscale) {
      for (int x = 0; x < pw; x++) {
	for (int c = 0; c < channels(); c++) {
	  p->SetValue(x, py, c, p->Value(x, py, c) + buf.Value(x, y, c));
	}
      }
      rem += 1;
    } else {
      float alpha = yscale - rem;
      for (int x = 0; x < pw; x++) {
	for (int c = 0; c < channels(); c++) {
	  p->SetValue(x, py, c, p->Value(x, py, c) + alpha * buf.Value(x, y, c));
	}
      }
      if (py < ph - 1) {
	for (int x = 0; x < pw; x++) {
	  for (int c = 0; c < channels(); c++) {
	    p->SetValue(x, py + 1, c, (1 - alpha) * buf.Value(x, y, c));
	  }
	}
      }
      py++;
      rem = 1 - alpha;
    }
  }

  // Average the values using the scaled area.
  for (int x = 0; x < pw; x++) {
    for (int y = 0; y < ph; y++) {
      for (int c = 0; c < channels(); c++) {
	p->SetValue(x, y, c, p->Value(x, y, c) / (xscale * yscale));
      }
    }
  }
}

void Patch::ExtractLabelInterp(const Label& l, Patch* p) const {
  int lw = l.w();
  int lh = l.h();
  int pw = p->width();
  int ph = p->height();
  int x0 = l.x();
  int y0 = l.y();
  float xscale = float(lw)/float(pw);
  float yscale = float(lh)/float(ph);

  for (int x = 0; x < pw; x++) {
    for (int y = 0; y < ph; y++) {
      float ix = (x + 0.5)*xscale;
      float iy = (y + 0.5)*yscale;
      int xa = int(floor(ix + x0 - 0.5));
      int ya = int(floor(iy + y0 - 0.5));
      int xb = int(ceil(ix + x0 - 0.5));
      int yb = int(ceil(iy + y0 - 0.5));

      xa = min(width()-1, max(0, xa));
      ya = min(height()-1, max(0, ya));
      xb = min(width()-1, max(0, xb));
      yb = min(height()-1, max(0, yb));

      float px = (xb - xa > 0) ? (ix + x0 - 0.5 - xa) / ((float)(xb - xa)) : 1.0;
      float py = (yb - ya > 0) ? (iy + y0 - 0.5 - ya) / ((float)(yb - ya)) : 1.0;
      for (int c = 0; c < channels(); c++) {
	float inter0 = (1.0-py)*Value(xa, ya, c) + (py)*Value(xa, yb, c);
	float inter1 = (1.0-py)*Value(xb, ya, c) + (py)*Value(xb, yb, c);
	p->SetValue(x, y, c, (1.0-px)*inter0 + (px)*inter1);
      }
    }
  }
}

void Patch::ExtractLabelNearest(const Label& l, Patch* p) const {
  int lw = l.w();
  int lh = l.h();
  int pw = p->width();
  int ph = p->height();
  int x0 = l.x();
  int y0 = l.y();
  float xscale = float(lw)/float(pw);
  float yscale = float(lh)/float(ph);

  for (int x = 0; x < pw; x++) {
    for (int y = 0; y < ph; y++) {
      float ix = (x + 0.5)*xscale;
      float iy = (y + 0.5)*yscale;
      int xn = (int)(ix + x0);  // Really ix + x0 - 0.5 + 0.5
      int yn = (int)(iy + y0);  // Really ix + y0 - 0.5 + 0.5

      xn = min(width()-1, max(0, xn));
      yn = min(height()-1, max(0, yn));

      for (int c = 0; c < channels(); c++) {
	p->SetValue(x, y, c, Value(xn, yn, c));
      }
    }
  }
}

void Patch::GenerateAllPatches(int width, int height, int step,
                               vector<Label>* labels, vector<Patch>* patches) const {
  for (int h = 0; h < this->height() - height; h += step) {
    for (int w = 0; w < this->width() - width; w += step) {
      Label l(w, h, width, height);

      Patch p(0, FLAGS_patch_width, FLAGS_patch_height, 1);
      ExtractLabel(l, &p);
      p.ComputeIntegralImage();

      labels->push_back(l);
      patches->push_back(p);
    }
  }
}

bool Patch::FromMessage(const PatchMessage& msg) {
  width_ = msg.width();
  height_ = msg.height();
  channels_ = msg.depth();
  label_ = msg.label();

  if (msg.data_size() != width_ * height_ * channels_)
    return false;

  data_.resize(width_ * height_ * channels_);
  copy(msg.data().begin(), msg.data().end(), data_.begin());

  return true;
}

void Patch::ToMessage(PatchMessage* msg) const {
  msg->set_width(width_);
  msg->set_height(height_);
  msg->set_depth(channels_);
  msg->set_label(label_);

  for (int i = 0; i < (int)(data_.size()); i++) {
    msg->add_data(data_[i]);
  }
}

bool Patch::Read(istream& in) {
  PatchMessage msg;
  
  return ReadMessage(in, &msg) && FromMessage(msg);
}

void Patch::Write(ostream& out) const {
  PatchMessage msg;
  
  ToMessage(&msg);
  WriteMessage(out, msg);
}

bool Patch::WritePPM(string filename) const {
  ofstream out(filename.c_str(), ofstream::out);
  
  out << "P6\n";
  out << width_ << " " << height_ << "\n";
  out << "255\n";
  if (channels_ == 3) {
    for (int h = 0; h < height_; h++) {
      for (int w = 0; w < width_; w++) {
	for (int c = 0; c < channels_; c++) {
	  unsigned char byte = 255.0 * Value(w,h,c);
	  out.write((char*)(&byte), sizeof(char));
	}
      }
    }
  } else {
    for (int h = 0; h < height_; h++) {
      for (int w = 0; w < width_; w++) {
	unsigned char byte = 255.0 * Value(w,h,0);
	out.write((char*)(&byte), sizeof(char));
	out.write((char*)(&byte), sizeof(char));
	out.write((char*)(&byte), sizeof(char));
      }
    }
  }
  out.close();
  
  return true;
}

bool Patch::WritePGM(string filename) const {
  ofstream out(filename.c_str(), ofstream::out);
  
  out << "P5\n";
  out << width_ << " " << height_ << "\n";
  out << "255\n";
  if (channels_ == 3) {
    for (int h = 0; h < height_; h++) {
      for (int w = 0; w < width_; w++) {
	float r = Value(w,h,0);
	float g = Value(w,h,1);
	float b = Value(w,h,2);
	unsigned char byte = 255 * (0.2989 * r + 0.5870 * g + 0.1140 * b);
	out.write((char*)(&byte), sizeof(char));
      }
    }
  } else {
    for (int h = 0; h < height_; h++) {
      for (int w = 0; w < width_; w++) {
	unsigned char byte = 255.0 * Value(w,h,0);
	out.write((char*)(&byte), sizeof(char));
      }
    }
  }
  out.close();
  
  return true;
}

}  //namespace speedboost
