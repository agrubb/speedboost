//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>

#include "detector.h"
#include "feature.h"
#include "patch.h"

using namespace std;

DEFINE_double(percentage, 0.2,
	      "Percentage of image to compute updates on.");
DEFINE_int32(num_trials, 500000,
	     "Number of performance trials to run.");
DEFINE_double(feature_limit, 1000.0,
	      "Maximum number of features (per pixel) to compute.");
DEFINE_double(merging_overlap, 0.5,
	      "Maximum amount detections can overlap and still be considered two "
              "different detections.  Given as a ratio of the overlapping area to "
              "the total area of the detection.");
DEFINE_bool(use_average_features, true,
	    "Use the average number of features per pixel, instead of the maximum.");

namespace speedboost {

Sequencer::Sequencer(Classifier* c)
  : c_(c) {
  next_biggest_.resize(c_->chains_.size());
  for (int i = 0; i < (int)(c_->chains_.size()); i++) {
    int next = -1;
    if (c_->filters_[i].active_) {
      for (int j = i + 1; j < (int)(c_->chains_.size()); j++) {
        if (!c_->filters_[j].active_) {
          break;
        }
        if (c_->filters_[j].active_ && (c_->filters_[j].threshold_ > c_->filters_[i].threshold_)) {
          next = j;
          break;
        }
      }
    }
    next_biggest_[i] = next;
  }
  
  max_threshold_.resize(c_->chains_.size());
  for (int i = 0; i < (int)(c_->chains_.size()); i++) {
    float max_thresh = c_->filters_[i].active_ ? c_->filters_[i].threshold_ : -1.0f;

    if (c_->filters_[i].active_) {
      for (int j = i + 1; j < (int)(c_->chains_.size()); j++) {
        max_thresh = max( max_thresh, c_->filters_[j].active_ ? c_->filters_[j].threshold_ : -1.0f );
        if (!c_->filters_[j].active_) {
          break;
        }
      }
    }

    max_threshold_[i] = max_thresh;
  }
}
  
int Sequencer::NextChain(int current_chain, float activation) const {
  if (activation > max_threshold_[current_chain])
    return -1;
  
  int next_chain = current_chain;
  while (next_chain > 0) {
    if (activation < c_->filters_[next_chain].threshold_) {
      return next_chain;
    } else {
      next_chain = next_biggest_[next_chain];
    }
  }

  return next_chain;
}

SingleScaleDetector::SingleScaleDetector(Classifier* c, Patch* integral)
  : c_(c), integral_(integral),
    chain_index_(0), stump_index_(0),
    default_indices_(),
    indices_(c->chains_.size()),
    num_pixels_((integral->height() - FLAGS_patch_height + 1) * (integral->width() - FLAGS_patch_width + 1)),
    updated_pixels_(0) {
  for (int h = 0; h < integral->height() - FLAGS_patch_height + 1; h++) {
    for (int w = 0; w < integral->width() - FLAGS_patch_width + 1; w++) {
      default_indices_.push_back(h * integral->width() + w);
    }
  }
}

void SingleScaleDetector::EvaluateAllPatches(float weight, const DecisionStump& stump, const Patch& frame, Patch* activations) {
  int pw = FLAGS_patch_width;
  int ph = FLAGS_patch_height;
  int fw = frame.width();
  int fh = frame.height();
  int aw = activations->width();
  // int ah = activations->height();

  // cout << "frame: " << fw << "x" << fh << endl;
  // cout << "patch: " << pw << "x" << ph << endl;
  // cout << "acts: " << activations->width() << "x" << activations->height() << endl;

  // c * width_ * height_ + h * width_ + w
  int p0 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y0_ * fw + stump.base_.b0_.x0_;
  int p1 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y1_ * fw + stump.base_.b0_.x0_;
  int p2 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y0_ * fw + stump.base_.b0_.x1_;
  int p3 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y1_ * fw + stump.base_.b0_.x1_;

  int p4 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y0_ * fw + stump.base_.b1_.x0_;
  int p5 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y1_ * fw + stump.base_.b1_.x0_;
  int p6 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y0_ * fw + stump.base_.b1_.x1_;
  int p7 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y1_ * fw + stump.base_.b1_.x1_;

  float output = stump.sign_ * weight;
  // cout << "activations: " << activations->data_.size() << endl;
  // cout << "frame: " << frame.data_.size() << endl;

  for (int ay = 0; ay < (fh - ph + 1); ay++) { 
    int fay = ay * fw;
    int aay = ay * aw;

    for (int ax = 0; ax < (fw - pw + 1); ax++) {
      // cout << ax << " " << ay << endl;
      // cout << aay + ax << " " << fay + ax + p0 << endl;
      // cout << frame.data_[fay + ax + p0] << " " << frame.data_[fay + ax + p1] << " "
      //      << frame.data_[fay + ax + p2] << " " << frame.data_[fay + ax + p3] << " "
      //      << frame.data_[fay + ax + p4] << " " << frame.data_[fay + ax + p5] << " "
      //      << frame.data_[fay + ax + p6] << " " << frame.data_[fay + ax + p7] << endl;
      // cout << ((frame.data_[fay + ax + p0] + frame.data_[fay + ax + p3]) -
      //          (frame.data_[fay + ax + p1] + frame.data_[fay + ax + p2])) << " "
      //      << ((frame.data_[fay + ax + p4] + frame.data_[fay + ax + p7]) -
      //          (frame.data_[fay + ax + p5] + frame.data_[fay + ax + p6])) << endl;

      float v = (stump.base_.w0_*((frame.data_[fay + ax + p0] + frame.data_[fay + ax + p3]) -
				  (frame.data_[fay + ax + p1] + frame.data_[fay + ax + p2]))
		 + stump.base_.w1_*((frame.data_[fay + ax + p4] + frame.data_[fay + ax + p7]) -
				    (frame.data_[fay + ax + p5] + frame.data_[fay + ax + p6])));
      activations->data_[aay + ax] += ((v < stump.split_) ? -output : output);
      // cout << "value: " << v << ", activation is now: " << activations->data_[aay + ax] << endl;
    }
  }
}

void SingleScaleDetector::EvaluateAllPatchesFiltered(float weight, const DecisionStump& stump, const Patch& frame,
						     const Filter& filter, Patch* activations) {
  int pw = FLAGS_patch_width;
  int ph = FLAGS_patch_height;
  int fw = frame.width();
  int fh = frame.height();
  int aw = activations->width();
  // int ah = activations->height();

  // cout << "frame: " << fw << "x" << fh << endl;
  // cout << "patch: " << pw << "x" << ph << endl;
  // cout << "acts: " << activations->width() << "x" << activations->height() << endl;

  // c * width_ * height_ + h * width_ + w
  int p0 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y0_ * fw + stump.base_.b0_.x0_;
  int p1 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y1_ * fw + stump.base_.b0_.x0_;
  int p2 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y0_ * fw + stump.base_.b0_.x1_;
  int p3 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y1_ * fw + stump.base_.b0_.x1_;

  int p4 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y0_ * fw + stump.base_.b1_.x0_;
  int p5 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y1_ * fw + stump.base_.b1_.x0_;
  int p6 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y0_ * fw + stump.base_.b1_.x1_;
  int p7 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y1_ * fw + stump.base_.b1_.x1_;

  float output = stump.sign_ * weight;
  // cout << "activations: " << activations->data_.size() << endl;
  // cout << "frame: " << frame.data_.size() << endl;

  for (int ay = 0; ay < (fh - ph + 1); ay++) {
    int fay = ay * fw;
    int aay = ay * aw;
    
    for (int ax = 0; ax < (fw - pw + 1); ax++) {
      // cout << ax << " " << ay << endl;
      // cout << aay + ax << " " << fay + ax + p0 << endl;
      if (abs(activations->data_[aay + ax]) < filter.threshold_) {
        float v = (stump.base_.w0_*((frame.data_[fay + ax + p0] + frame.data_[fay + ax + p3]) -
                                    (frame.data_[fay + ax + p1] + frame.data_[fay + ax + p2]))
                   + stump.base_.w1_*((frame.data_[fay + ax + p4] + frame.data_[fay + ax + p7]) -
                                      (frame.data_[fay + ax + p5] + frame.data_[fay + ax + p6])));
        activations->data_[aay + ax] += ((v < stump.split_) ? -output : output);
      }
    }
  }
}

void SingleScaleDetector::EvaluateAllPatchesListed(float weight, const DecisionStump& stump, const Patch& frame,
						   const vector<int>& indices, Patch* activations) {
  // int pw = FLAGS_patch_width;
  // int ph = FLAGS_patch_height;
  int fw = frame.width();
  int fh = frame.height();
  // int aw = activations->width();
  // int ah = activations->height();

  // cout << "frame: " << fw << "x" << fh << endl;
  // cout << "patch: " << pw << "x" << ph << endl;
  // cout << "acts: " << activations->width() << "x" << activations->height() << endl;

  // c * width_ * height_ + h * width_ + w
  int p0 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y0_ * fw + stump.base_.b0_.x0_;
  int p1 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y1_ * fw + stump.base_.b0_.x0_;
  int p2 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y0_ * fw + stump.base_.b0_.x1_;
  int p3 = stump.base_.c_ * fw * fh  + stump.base_.b0_.y1_ * fw + stump.base_.b0_.x1_;

  int p4 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y0_ * fw + stump.base_.b1_.x0_;
  int p5 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y1_ * fw + stump.base_.b1_.x0_;
  int p6 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y0_ * fw + stump.base_.b1_.x1_;
  int p7 = stump.base_.c_ * fw * fh  + stump.base_.b1_.y1_ * fw + stump.base_.b1_.x1_;

  float output = stump.sign_ * weight;
  // cout << "activations: " << activations->data_.size() << endl;
  // cout << "frame: " << frame.data_.size() << endl;

  for (int i = 0; i < (int)(indices.size()); i++) {
    int idx = indices[i];
    // cout << ax << " " << ay << endl;
    // cout << aay + ax << " " << fay + ax + p0 << endl;
    float v = (stump.base_.w0_*((frame.data_[idx + p0] + frame.data_[idx + p3]) -
				(frame.data_[idx + p1] + frame.data_[idx + p2]))
	       + stump.base_.w1_*((frame.data_[idx + p4] + frame.data_[idx + p7]) -
				  (frame.data_[idx + p5] + frame.data_[idx + p6])));
    activations->data_[idx] += ((v < stump.split_) ? -output : output);
  }
}

bool SingleScaleDetector::HasMoreFeatures() {
  return (chain_index_ < (int)(c_->chains_.size())) && (stump_index_ < (int)(c_->chains_[chain_index_].stumps_.size()));
}

void SingleScaleDetector::ComputeNextFeature(const Sequencer& sequencer,
					     Patch* activations, Patch* updates) {
  // cout << "*** chain_index_: " << chain_index_ << " stump_index_: " << stump_index_ << endl;
  if (!HasMoreFeatures())
    return;

  // Update the activations with the next feature.
  if (c_->filters_[chain_index_].active_) {
    if ((c_->type_ == Classifier::kCascade) && (stump_index_ == 0) && (chain_index_ > 0)) {
      for (int i = 0; i < (int)(indices_[chain_index_].size()); i++) {
	int idx = indices_[chain_index_][i];
	activations->data_[idx] = 0.0;
      }
    }

    EvaluateAllPatchesListed(c_->chains_[chain_index_].weights_[stump_index_],
                             c_->chains_[chain_index_].stumps_[stump_index_], *integral_,
			     indices_[chain_index_], activations);

    updated_pixels_ += indices_[chain_index_].size();
    if (updates) {
      for (int i = 0; i < (int)(indices_[chain_index_].size()); i++) {
	updates->data_[indices_[chain_index_][i]] += 1.0;
      }
    }
  } else {
    EvaluateAllPatches(c_->chains_[chain_index_].weights_[stump_index_],
                       c_->chains_[chain_index_].stumps_[stump_index_],
                       *integral_, activations);

    updated_pixels_ += num_pixels_;
    if (updates) {
      for (int i = 0; i < (int)(default_indices_.size()); i++) {
	updates->data_[default_indices_[i]] += 1.0;
      }
    }
  }

  stump_index_++;
  if (stump_index_ == (int)(c_->chains_[chain_index_].stumps_.size())) {
    chain_index_++;
    stump_index_ = 0;

    if (chain_index_ < (int)(c_->chains_.size())) {
      vector<int>* inds = &default_indices_;
      if (c_->filters_[chain_index_ - 1].active_) {
	inds = &(indices_[chain_index_ - 1]);
      }

      if (c_->type_ == Classifier::kCascade) {
	for (int i = 0; i < (int)(inds->size()); i++) {
	  if (activations->data_[(*inds)[i]] > c_->filters_[chain_index_].threshold_)
	    indices_[chain_index_].push_back((*inds)[i]);
	}
      } else if (c_->type_ == Classifier::kAnytime) {
	if (c_->filters_[chain_index_].active_) {
	  for (int i = 0; i < (int)(inds->size()); i++) {
	    float v = abs(activations->data_[(*inds)[i]]);
            int next = sequencer.NextChain(chain_index_, v);

	    if (next > 0)
              indices_[next].push_back((*inds)[i]);
          }
        }
      }
    }

    // Try and delete old index data we don't need anymore.
    if (c_->filters_[chain_index_ - 1].active_) {
      vector<int>().swap(indices_[chain_index_ - 1]);
    }
  }

  // cout << "indices_ sizes:  ";
  // for (int i = 0; i < indices_.size(); i++) {
  //   cout << " " << indices_[i].size();
  // }
  // cout << endl;
}

Detector::Detector(Classifier* c)
  : c_(c), sequencer_(c) {
}

void Detector::SetupForFrame(const Patch& frame, int num_scales, float scaling_factor,
                             vector<Patch>* scaled_integrals, vector<Patch>* scaled_activations,
                             vector<SingleScaleDetector>* scaled_detectors) {
  scaled_integrals->clear();
  scaled_activations->clear();
  scaled_detectors->clear();

  float current_scale = 1.0;
  for (int i = 0; i < num_scales; i++) {
    Patch integral(0, frame.width()*current_scale, frame.height()*current_scale, 1);
    Patch activations(0, frame.width()*current_scale, frame.height()*current_scale, 1);
    Label l(0, 0, frame.width(), frame.height());

    frame.ExtractLabel(l, &integral);
    integral.ComputeIntegralImage();

    scaled_integrals->push_back(integral);
    scaled_activations->push_back(activations);
    current_scale = current_scale / scaling_factor;
  }

  // Make these after to avoid memory issues.
  for (int i = 0; i < num_scales; i++) {
    scaled_detectors->push_back(SingleScaleDetector(c_, &(*scaled_integrals)[i]));
  }
}

void Detector::ComputeActivationPyramid(const Patch& frame, int num_scales, float scaling_factor,
                                        vector<Patch>* scaled_activations) {
  vector<Patch> scaled_integrals;
  vector<SingleScaleDetector> scaled_detectors;
  SetupForFrame(frame, num_scales, scaling_factor,
                &scaled_integrals, scaled_activations, &scaled_detectors);

  Tic();

  float features_computed = 0;
  float frame_index = 0;
  while (scaled_detectors[0].HasMoreFeatures() && (features_computed < FLAGS_feature_limit)) {
    for (int i = 0; i < (int)(scaled_detectors.size()); i++) {
      scaled_detectors[i].ComputeNextFeature(sequencer_, &((*scaled_activations)[i]));
    }

    if (FLAGS_use_average_features) {
      features_computed = 0;
      for (int i = 0; i < (int)(scaled_detectors.size()); i++) {
        features_computed += scaled_detectors[i].FeaturesPerPixel() / (float)num_scales;
      }
    } else {
      features_computed++;
    }
    frame_index++;
  }

  cout << "Time elapsed: " << Toc() << endl;
  cout << "Total features computed: " << features_computed << " in " << frame_index << " frames." << endl;
}

void OutputActivation(const Patch& activations, string filename) {
  Patch p(activations);

  for (int h = 0; h < p.height(); h++) {
    for (int w = 0; w < p.width(); w++) {
      float v = p.Value(w,h,0);
      p.SetValue(w,h,0,exp(v) / (1.0 + exp(v)));
    }
  }

  p.WritePGM(filename);
}

void Detector::ComputeMergedActivation(const Patch& frame, int num_scales, float scaling_factor, Patch* merged) {
  vector<Patch> activations;
  ComputeActivationPyramid(frame, num_scales, scaling_factor, &activations);

  Patch inflated(0, frame.width(), frame.height(), 1);
  *merged = Patch(0, frame.width(), frame.height(), 1);
  for (int h = 0; h < frame.height(); h++) {
    for (int w = 0; w < frame.width(); w++) {
      merged->SetValue(w, h, 0, -FLT_MAX);
    }
  }
    
  for (int i = 0; i < (int)(activations.size()); i++) {
    // First shift image so border is centered.
    Patch shifted(0, activations[i].width(), activations[i].height(), 1);
    for (int h = 0; h < shifted.height(); h++) {
      for (int w = 0; w < shifted.width(); w++) {
        shifted.SetValue(w, h, 0, -FLT_MAX);
      }
    }
    float hborder = (FLAGS_patch_height + 1) / 2;
    float wborder = (FLAGS_patch_width + 1) / 2;
    for (int h = 0; h < shifted.height() - FLAGS_patch_height + 1; h++) {
      for (int w = 0; w < shifted.width() - FLAGS_patch_width + 1; w++) {
        shifted.SetValue(w + wborder, h + hborder, 0, activations[i].Value(w, h, 0));
      }
    }

    Label l(0, 0, activations[i].width(), activations[i].height());
    // Resize image using nearest neighbor
    shifted.ExtractLabel(l, &inflated, true);

    stringstream ss;
    ss << "tmp/shifted." << i << ".pgm";
    string filename = ss.str();
    OutputActivation(shifted, filename);

    for (int h = 0; h < frame.height(); h++) {
      for (int w = 0; w < frame.width(); w++) {
        merged->SetValue(w, h, 0, max(merged->Value(w, h, 0),
                                      inflated.Value(w, h, 0)));
      }
    }
  }
}

void Detector::ComputeDetections(const Patch& frame, int num_scales, float scaling_factor,
                                 float detection_threshold, vector<Label>* detections) {
  vector<Patch> activations;
  ComputeActivationPyramid(frame, num_scales, scaling_factor, &activations);

  vector<Label> all_detections;
  vector<float> all_weights;

  float current_scale = 1.0;
  for (int i = 0; i < (int)(activations.size()); i++) {
    for (int h = 0; h < activations[i].height(); h++) {
      for (int w = 0; w < activations[i].width(); w++) {
        if (activations[i].Value(w,h,0) > detection_threshold) {
          Label l(w*current_scale, h*current_scale, FLAGS_patch_width*current_scale, FLAGS_patch_height*current_scale);
          all_detections.push_back(l);
          all_weights.push_back(activations[i].Value(w,h,0));
        }
      }
    }

    current_scale = current_scale * scaling_factor;
  }

  FilterDetections(all_detections, all_weights, FLAGS_merging_overlap, detections);
}

void Detector::FilterDetections(const vector<Label>& detections, const vector<float>& weights,
                                float overlap, vector<Label>* filtered) {
  cout << "Filtering detections..." << endl;
  cout << "Starting with " << detections.size() << " detections." << endl;

  vector< pair<float, int> > sortable(detections.size());
  for(int i = 0; i < (int)(detections.size()); i++) {
    sortable[i].first = weights[i];
    sortable[i].second = i;
    
    sort(sortable.begin(), sortable.end());
  }

  for (int i = (int)(detections.size()) - 1; i >= 0; i--) {
    bool passed = true;
    for (int j = 0; j < (int)(filtered->size()); j++) {
      // Check overlap between candidate and already selected detection.
      int dx = detections[i].x();
      int dy = detections[i].y();
      int dw = detections[i].w();
      int dh = detections[i].h();
      int fx = (*filtered)[j].x();
      int fy = (*filtered)[j].y();
      int fw = (*filtered)[j].w();
      int fh = (*filtered)[j].h();

      int x1 = max(dx, fx);
      int y1 = max(dy, fy);
      int x2 = min(dx + dw, fx + fw);
      int y2 = min(dy + dh, fy + fh);
      
      int w = max(0, x2 - x1);
      int h = max(0, y2 - y1);

      if (w * h > overlap * (float)(dw * dh)) {
        passed = false;
        break;
      }
    }
    
    if (passed) {
      filtered->push_back(detections[i]);
    }
  }
  cout << "Finished with " << filtered->size() << " detections." << endl;
}
}  // namespace speedboost
