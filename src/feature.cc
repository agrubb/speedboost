#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <string>
#include <stdlib.h>
#include <utility>
#include <vector>

#include "feature.pb.h"
#include "feature.h"
#include "patch.h"
#include "util.h"

using namespace std;

namespace speedboost {

bool Box::FromMessage(const BoxMessage& msg) {
  x0_ = msg.x0();
  y0_ = msg.y0();
  x1_ = msg.x1();
  y1_ = msg.y1();

  if (x0_ < 0 || x0_ >= FLAGS_patch_width ||
      x1_ < 0 || x1_ >= FLAGS_patch_width ||
      y0_ < 0 || y0_ >= FLAGS_patch_height ||
      y1_ < 0 || y1_ >= FLAGS_patch_height) {
    return false;
  }

  return true;
}

void Box::ToMessage(BoxMessage* msg) const {
  msg->set_x0(x0_);
  msg->set_y0(y0_);
  msg->set_x1(x1_);
  msg->set_y1(y1_);
}

bool Box::Read(istream& in) {
  BoxMessage msg;
  
  return ReadMessage(in, &msg) && FromMessage(msg);
}

void Box::Write(ostream& out) const {
  BoxMessage msg;
  
  ToMessage(&msg);
  WriteMessage(out, msg);
}

float Feature::Evaluate(const Patch& p) const {
  return (w0_*((p.Value(b0_.x0_, b0_.y0_, c_) + p.Value(b0_.x1_, b0_.y1_, c_)) -
	       (p.Value(b0_.x0_, b0_.y1_, c_) + p.Value(b0_.x1_, b0_.y0_, c_)))
	  + w1_*((p.Value(b1_.x0_, b1_.y0_, c_) + p.Value(b1_.x1_, b1_.y1_, c_)) -
		 (p.Value(b1_.x0_, b1_.y1_, c_) + p.Value(b1_.x1_, b1_.y0_, c_))));
}

bool Feature::FromMessage(const FeatureMessage& msg) {
  if (msg.type() != FeatureMessage::HAAR)
    return false;

  if (!msg.has_haar_data())
    return false;

  const HaarFeatureMessage& h_msg = msg.haar_data();
  c_ = h_msg.channel();

  if (c_ < 0 || c_ >= FLAGS_patch_depth) {
    return false;
  }

  if (!h_msg.has_b0() || !h_msg.has_w0())
    return false;
  b0_.FromMessage(h_msg.b0());
  w0_ = h_msg.w0();

  if (!h_msg.has_b1() || !h_msg.has_w1())
    return false;
  b1_.FromMessage(h_msg.b1());
  w1_ = h_msg.w1();

  return true;
}

void Feature::ToMessage(FeatureMessage* msg) const {
  msg->set_type(FeatureMessage::HAAR);
    
  HaarFeatureMessage* h_msg = msg->mutable_haar_data();
  h_msg->set_channel(c_);

  b0_.ToMessage(h_msg->mutable_b0());
  h_msg->set_w0(w0_);

  b1_.ToMessage(h_msg->mutable_b1());
  h_msg->set_w1(w1_);
}

bool Feature::Read(istream& in) {
  FeatureMessage msg;
  
  return ReadMessage(in, &msg) && FromMessage(msg);
}

void Feature::Write(ostream& out) const {
  FeatureMessage msg;
  
  ToMessage(&msg);
  WriteMessage(out, msg);
}

int Feature::ReadFeaturesFromFile(const string& filename, vector<Feature>* features)
{
  ifstream in(filename.c_str(), ifstream::in);
  int num_read = 0;
  while (in.good()) {
    Feature f;
    if (f.Read(in)) {
      features->push_back(f);
      num_read++;
    }
  }
  in.close();

  return num_read;
}

void Feature::WriteFeaturesToFile(const string& filename, const vector<Feature>& features)
{
  ofstream out(filename.c_str(), ofstream::out);
  for (unsigned int i = 0; i < features.size(); i++) {
    features[i].Write(out);
  }
  out.close();
}

void Feature::GenerateFeatures(int n, vector<Feature>* features)
{
  srand(time(NULL));
  for (int i = 0; i < n; i++) {
    int x0, y0, x1, y1;
    x0 = rand()%(FLAGS_patch_width - 2);
    x1 = rand()%(FLAGS_patch_width - x0 - 2) + x0 + 2;
    y0 = rand()%(FLAGS_patch_height - 2);
    y1 = rand()%(FLAGS_patch_height - y0 - 2) + y0 + 2;
    Box b0(x0, y0, x1, y1);

    x0 = rand()%(FLAGS_patch_width - 2);
    x1 = rand()%(FLAGS_patch_width - x0 - 2) + x0 + 2;
    y0 = rand()%(FLAGS_patch_height - 2);
    y1 = rand()%(FLAGS_patch_height - y0 - 2) + y0 + 2;
    Box b1(x0, y0, x1, y1);

    Feature f(b0, b1, 1.0, (rand()%2)*2 - 1, rand() % FLAGS_patch_depth);

    features->push_back(f);
  }
}

}  // namespace speedboost
