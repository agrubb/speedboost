//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "classifier.h"
#include "classifier.pb.h"
#include "feature.h"
#include "feature_selector.h"
#include "patch.h"
#include "util.h"

using namespace std;

DEFINE_bool(anytime_boost, false, "Run the anytime boosting algorithm (SpeedBoost).");
DEFINE_int32(max_inner_stages, 100, "Maximum number of inner stages in a chain for cascade.");
DEFINE_int32(stage_increment, 20, "Number of stages between resampling of new training data.");
DEFINE_double(target_false_negative, 0.005, "Desired false negative rate per casacade stage.");
DEFINE_double(target_false_positive_base, 0.85, "Desired false positive rate per casacade stage (base).");
DEFINE_double(target_false_positive_step, 0.05, "Desired false positive rate per casacade stage (step).");
DEFINE_bool(sample_patches, false, "Sample the loaded patches using the gradient as a weighted sample.");

namespace speedboost {

float DecisionStump::Evaluate(const Patch& p) const
{
  return Evaluate(base_.Evaluate(p));
}

float DecisionStump::Evaluate(float response) const
{
  // cout << "response: " << response << endl;
  return (response < split_) ? -sign_ : sign_;
}

bool DecisionStump::FromMessage(const StumpMessage& msg) {
  if (!base_.FromMessage(msg.base()))
    return false;

  if (!msg.has_split() || !msg.has_output())
    return false;

  split_ = msg.split();
  sign_ = msg.output();

  return true;
}

void DecisionStump::ToMessage(StumpMessage* msg) const {
  base_.ToMessage(msg->mutable_base());
  msg->set_split(split_);
  msg->set_output(sign_);
}

bool Filter::FromMessage(const FilterMessage& msg) {
  if (!msg.has_threshold() || !msg.has_active() || !msg.has_less())
    return false;

  threshold_ = msg.threshold();
  active_ = msg.active();
  less_ = msg.less();

  return true;
}

void Filter::ToMessage(FilterMessage* msg) const {
  msg->set_threshold(threshold_);
  msg->set_active(active_);
  msg->set_less(less_);
}

bool Chain::FromMessage(const ChainMessage& msg) {
  stumps_.resize(msg.stumps_size());
  weights_.resize(msg.stumps_size());
  biases_.resize(msg.stumps_size());
  for (int i = 0; i < msg.stumps_size(); i++) {
    if (!stumps_[i].FromMessage(msg.stumps(i).stump()))
      return false;

    weights_[i] = msg.stumps(i).weight();
    biases_[i] = msg.stumps(i).bias();
  }

  return true;
}

void Chain::ToMessage(ChainMessage* msg) const {
  for (int i = 0; i < (int)(stumps_.size()); i++) {
    WeightedStumpMessage* ws_msg = msg->add_stumps();
    stumps_[i].ToMessage(ws_msg->mutable_stump());
    ws_msg->set_weight(weights_[i]);
    ws_msg->set_bias(biases_[i]);
  }
}

bool Chain::Read(istream& in) {
  ChainMessage msg;
  
  return ReadMessage(in, &msg) && FromMessage(msg);
}

void Chain::Write(ostream& out) const {
  ChainMessage msg;
  
  ToMessage(&msg);
  WriteMessage(out, msg);
}

bool Classifier::FromMessage(const ClassifierMessage& msg) {
  if (msg.type() == ClassifierMessage::BOOSTED) {
    type_ = kBoosted;
  } else if (msg.type() == ClassifierMessage::CASCADE) {
    type_ = kCascade;
    filters_use_margin_ = false;
    filters_are_additive_ = false;
    filters_are_permanent_ = true;  
  } else if (msg.type() == ClassifierMessage::ANYTIME) {
    type_ = Classifier::kAnytime;
    filters_use_margin_ = true;
    filters_are_additive_ = true;
    filters_are_permanent_ = false;
  }

  chains_.resize(msg.chains_size());
  filters_.resize(msg.chains_size());
  for (int i = 0; i < msg.chains_size(); i++) {
    if (!chains_[i].FromMessage(msg.chains(i).chain()))
      return false;

    if (!filters_[i].FromMessage(msg.chains(i).filter()))
      return false;
  }

  if (msg.has_patch_width()) {
    if (msg.patch_width() != FLAGS_patch_width) {
      if (google::GetCommandLineFlagInfoOrDie("patch_width").is_default) {
        cout << "WARNING: changing patch_width flag from default of " << FLAGS_patch_width
             << " to " << msg.patch_width() << " to match input classifier." << endl;
        FLAGS_patch_width = msg.patch_width();
      } else {
        cout << "ERROR: patch_width specified in flags differs from patch_width classifier was trained with" << endl;
        return false;
      }
    }
  }

  if (msg.has_patch_height()) {
    if (msg.patch_height() != FLAGS_patch_height) {
      if (google::GetCommandLineFlagInfoOrDie("patch_height").is_default) {
        cout << "WARNING: changing patch_height flag from default of " << FLAGS_patch_height
             << " to " << msg.patch_height() << " to match input classifier." << endl;
        FLAGS_patch_height = msg.patch_height();
      } else {
        cout << "ERROR: patch_height specified in flags differs from patch_height classifier was trained with" << endl;
        return false;
      }
    }
  }

  if (msg.has_patch_depth()) {
    if (msg.patch_depth() != FLAGS_patch_depth) {
      if (google::GetCommandLineFlagInfoOrDie("patch_depth").is_default) {
        cout << "WARNING: changing patch_depth flag from default of " << FLAGS_patch_depth
             << " to " << msg.patch_depth() << " to match input classifier." << endl;
        FLAGS_patch_depth = msg.patch_depth();
      } else {
        cout << "ERROR: patch_depth specified in flags differs from patch_depth classifier was trained with" << endl;
        return false;
      }
    }
  }

  return true;
}

void Classifier::ToMessage(ClassifierMessage* msg) const {
  if (type_ == kBoosted) msg->set_type(ClassifierMessage::BOOSTED);
  if (type_ == kCascade) msg->set_type(ClassifierMessage::CASCADE);
  if (type_ == kAnytime) msg->set_type(ClassifierMessage::ANYTIME);

  for (int i = 0; i < (int)(chains_.size()); i++) {
    FilteredChainMessage* fc_msg = msg->add_chains();
    chains_[i].ToMessage(fc_msg->mutable_chain());
    filters_[i].ToMessage(fc_msg->mutable_filter());
  }

  msg->set_patch_width(FLAGS_patch_width);
  msg->set_patch_height(FLAGS_patch_height);
  msg->set_patch_depth(FLAGS_patch_depth);
}

bool Classifier::Read(istream& in) {
  ClassifierMessage msg;
  
  return ReadMessage(in, &msg) && FromMessage(msg);
}

void Classifier::Write(ostream& out) const {
  ClassifierMessage msg;
  
  ToMessage(&msg);
  WriteMessage(out, msg);
}

bool Classifier::ReadFromFile(const std::string& filename) {
  ClassifierMessage msg;
  
  return ReadMessageFromFileAsText(filename, &msg) && FromMessage(msg);
}

bool Classifier::WriteToFile(const std::string& filename) const {
  ClassifierMessage msg;
  
  ToMessage(&msg);
  return WriteMessageToFileAsText(filename, msg);
}

void Classifier::Print() const {
  cout << endl;
  for (unsigned int i = 0; i < chains_.size(); i++) {
    cout << "Stage " << i << endl;
    cout <<         "-------------" << endl;
    
    filters_[i].Print();

    cout << "Features:" << endl;
    for (unsigned int j = 0; j < chains_[i].stumps_.size(); j++) { 
      cout << "  " << chains_[i].weights_[j] << "  *  ";
      chains_[i].stumps_[j].Print();
      //cout << "bias: " << chains_[i].biases_[j] << endl;
    }
    
    cout << endl << endl;
  }
}

float ZeroOneLoss(const vector<Patch> &patches, const vector<float> activations,
                  float* positive_loss, float* negative_loss) {
  float loss = 0;
  int positive_count = 0;
  int negative_count = 0;
  *positive_loss = 0;
  *negative_loss = 0;

  for (unsigned int i = 0; i < patches.size(); i++) {
    int y = (patches[i].label() > 0) ? 1.0 : -1.0;
    int sign = (activations[i] > 0) ? 1.0 : -1.0;
    if (y != sign) {
      loss += 1;
    }

    if (patches[i].label() > 0) {
      positive_count++;
      if (y != sign) {
        *positive_loss += 1.0;
      }
    } else {
      negative_count++;
      if (y != sign) {
        *negative_loss += 1.0;
      }
    }
  }

  *positive_loss = *positive_loss / positive_count;
  *negative_loss = *negative_loss / negative_count;

  return loss / ((float)patches.size());
}


float ZeroOneLoss(const vector<Patch>& patches,
		  const vector<float>& sample_weights,
		  const vector<float> activations,
                  float* positive_loss, float* negative_loss) {
  if (sample_weights.size() != activations.size())
    return ZeroOneLoss(patches, activations, positive_loss, negative_loss);
  
  float loss = 0;
  float count = 0;
  float positive_count = 0;
  float negative_count = 0;
  *positive_loss = 0;
  *negative_loss = 0;

  for (unsigned int i = 0; i < patches.size(); i++) {
    int y = (patches[i].label() > 0) ? 1.0 : -1.0;
    int sign = (activations[i] > 0) ? 1.0 : -1.0;
    if (y != sign) {
      loss += sample_weights[i];
    }
    
    if (patches[i].label() > 0) {
      positive_count += sample_weights[i];
      if (y != sign) {
        *positive_loss += sample_weights[i];
      }
    } else {
      negative_count += sample_weights[i];
      if (y != sign) {
        *negative_loss += sample_weights[i];
      }
    }
    count += sample_weights[i];
  }
  
  *positive_loss = *positive_loss / positive_count;
  *negative_loss = *negative_loss / negative_count;
  
  return loss / count;
}

float ExpLoss(const vector<Patch>& patches,
	      const vector<float>& activations) {
  float loss = 0;
  
  // if (sample_weights.size() == activations.size()) {
  //   for (unsigned int i = 0; i < patches.size(); i++) {
  // 	float y = (patches[i].label() > 0) ? 1.0 : -1.0;
  // 	loss += sample_weights[i] * exp(- y * activations[i]);
  //   }
  // } else {
  for (unsigned int i = 0; i < patches.size(); i++) {
    float y = (patches[i].label() > 0) ? 1.0 : -1.0;
    loss += exp(- y * activations[i]);
  }
  //    }
  return loss;
}

float ExpLoss(const vector<Patch>& patches,
	      const vector<float>& sample_weights,
	      const vector<float>& activations) {
  float loss = 0;

  if (sample_weights.size() == activations.size()) {
    for (unsigned int i = 0; i < patches.size(); i++) {
      float y = (patches[i].label() > 0) ? 1.0 : -1.0;
      loss += sample_weights[i] * exp(- y * activations[i]);
    }
  } else {
    for (unsigned int i = 0; i < patches.size(); i++) {
      float y = (patches[i].label() > 0) ? 1.0 : -1.0;
      loss += exp(- y * activations[i]);
    }
  }
  return loss;
}

void Gradient(const vector<Patch> &patches, const vector<float>& sample_weights,
	      const vector<float> activations, vector<float>* weights) {
  if (sample_weights.size() == activations.size()) {
    cout << "Using sample weights to re-weight gradient..." << endl;
    for (unsigned int i = 0; i < activations.size(); i++) {
      float y = (patches[i].label() > 0) ? 1.0 : -1.0;
      (*weights)[i] = sample_weights[i] * exp( -y * activations[i]);
    }
  } else {
    for (unsigned int i = 0; i < activations.size(); i++) {
      float y = (patches[i].label() > 0) ? 1.0 : -1.0;
      (*weights)[i] = exp( -y * activations[i]);
    }
  }
}


float Activation(const Patch& patch, const Classifier& c) {
  float activation = 0;
  for (unsigned int i = 0; i < c.chains_.size(); i++) {
    float v = (c.filters_use_margin_) ? abs(activation) : activation;
    if (c.filters_[i].PassesFilter(v)) {
      if (c.filters_[i].active_ && !c.filters_are_additive_) {
	activation = 0.0;
      }

      for (unsigned int j = 0; j < c.chains_[i].stumps_.size(); j++) {
	activation += c.chains_[i].weights_[j] * c.chains_[i].stumps_[j].Evaluate(patch);
      }
    } else {
      if (c.filters_are_permanent_) {
        break;
      }
    }
  }
  
  return activation;
}

float Classifier::Activation(const Patch& patch) const {
  // cout << "Activation():" << endl;
  float activation = 0;
  for (unsigned int i = 0; i < chains_.size(); i++) {
    float v = (filters_use_margin_) ? abs(activation) : activation;
    if (filters_[i].PassesFilter(v)) {
      if (filters_[i].active_ && !filters_are_additive_) {
	activation = 0.0;
      }

      for (unsigned int j = 0; j < chains_[i].stumps_.size(); j++) {
	activation += chains_[i].weights_[j] * chains_[i].stumps_[j].Evaluate(patch);
        // cout << i << " " << j << " act: " << activation << endl;
      }
    } else {
      if (filters_are_permanent_) {
        break;
      }
    }
  }
  
  return activation;
}

bool Classifier::IsActiveInLastChain(const Patch& patch) const {
  float activation = 0;
  bool active = true;
  for (unsigned int i = 0; i < chains_.size(); i++) {
    float v = (filters_use_margin_) ? abs(activation) : activation;
    if (filters_[i].PassesFilter(v)) {
      active = true;
      if (filters_[i].active_ && !filters_are_additive_) {
	activation = 0.0;
      }

      for (unsigned int j = 0; j < chains_[i].stumps_.size(); j++) {
	activation += chains_[i].weights_[j] * chains_[i].stumps_[j].Evaluate(patch);
      }
    } else {
      active = false;
      if (filters_are_permanent_) {
	break;
      }
    }
  }
  
  return active;
}

float ComputePredictionBias(const vector<Patch> &patches, const vector<float> activations,
                            float false_negative_rate, float* false_positive_rate) {
  vector< pair<float, char> > sortable(patches.size());
  for (unsigned int p = 0; p < patches.size(); p++) {
    sortable[p].first = activations[p];
    sortable[p].second = patches[p].label();
  }

  sort(sortable.begin(), sortable.end());

  int positives = 0;
  int negatives = 0;

  for (unsigned int p = 0; p < patches.size(); p++) {
    if (sortable[p].second > 0) {
      positives++;
    } else {
      negatives++;
    }
  }

  float false_negatives = 0;
  float false_positives = negatives;

  float bias = 0.0;
  *false_positive_rate = 1.0;

  for (unsigned int p = 0; p < patches.size(); p++) {
    if (sortable[p].second > 0) {
      false_negatives++;
    } else {
      false_positives--;
    }

    if (sortable[p].first == sortable[p + 1].first) continue;

    if (false_negatives / ((float)positives) > false_negative_rate) {
      break;
    } else {
      bias = (sortable[p].first + sortable[p + 1].first) / 2.0;
      *false_positive_rate = (false_positives / ((float)negatives));
    }
  }

  return bias;
}

void TrainStages(const vector<Patch>& patches, const vector<float>& sample_weights,
		 const vector<Feature>& features,
		 int max_num_stages, bool calc_weights, bool use_rates,
		 float false_negative_rate, float false_positive_rate,
		 const vector<Patch>& validation, Classifier* c) {
  FeatureSelector selector(patches, features);

  vector<float> weights(patches.size(), 1.0);
  vector<float> activations(patches.size(), 0.0);

  vector<float> validation_activations(validation.size(), 0.0);

  float positive_loss, negative_loss;

  Chain *last_chain = &(c->chains_.back());

  // If anytime boosting, don't throw away gradient.
  if (calc_weights) {
    for (unsigned int p = 0; p < patches.size(); p++) {
      activations[p] = Activation(patches[p], *c);
    }
    Gradient(patches, sample_weights, activations, &weights);
  }

  cout << "Initial" << endl;
  cout << "exp loss: " << ExpLoss(patches, sample_weights, activations)
       << ", 0/1 loss: " << ZeroOneLoss(patches, sample_weights, activations,
                                        &positive_loss, &negative_loss) << endl;
  cout << "+ err: " << positive_loss << ", - err: " << negative_loss << endl;

  float bias;
  for (int i = 0; i < max_num_stages; i++) {
    int index;
    float err;

    // cout << "weights: " << endl;
    // for (unsigned int j = 0; j < weights.size(); j++) {
    //   cout << " " << weights[j];
    // }
    // cout << endl << endl;

    // cout << "sample_weights: " << endl;
    // for (unsigned int j = 0; j < sample_weights.size(); j++) {
    //   cout << " " << sample_weights[j];
    // }
    // cout << endl << endl;
    
    DecisionStump feat;
    Filter filt;
    if (FLAGS_anytime_boost) {
      feat = selector.SelectFeatureAndThreshold(weights, activations, &index, &err, &(filt.threshold_));
      if (filt.threshold_ < FLT_MAX) {
	filt.active_ = true;
      }
      filt.less_ = true;
    } else {
      feat = selector.SelectFeature(weights, activations, &index, &err);
    }

    float alpha = 0.5 * log( (1 - err) / err );

    float fpr;
    bias = ComputePredictionBias(validation, validation_activations, false_negative_rate, &fpr);

    last_chain->stumps_.push_back(feat);
    last_chain->weights_.push_back(alpha);
    last_chain->biases_.push_back(bias);

    if (FLAGS_anytime_boost) {
      // Set filter and add a new chain for the next feature.
      c->filters_.back() = filt;
      
      c->chains_.push_back(Chain());
      c->filters_.push_back(Filter());
      last_chain = &(c->chains_.back());
    }

    selector.UpdateActivations(feat, filt, index, alpha, &activations);
    Gradient(patches, sample_weights, activations, &weights);

    cout << endl << "Iteration " << i << endl;
    cout <<         "-------------" << endl;

    cout << endl << "Selected feature:" << endl;
    feat.Print();

    cout << endl << "Selected filter:" << endl;
    filt.Print();

    cout << endl << "alpha: " << alpha << endl;

    cout << endl;
    cout << "exp loss: " << ExpLoss(patches, sample_weights, activations)
         << ", 0/1 loss: " << ZeroOneLoss(patches, sample_weights, activations,
                                          &positive_loss, &negative_loss) << endl;
    cout << "+ err: " << positive_loss << ", - err: " << negative_loss << endl;

    cout << "validation activations: " << validation_activations.size() << endl;
    for (unsigned int p = 0; p < validation.size(); p++) {
      validation_activations[p] = Activation(validation[p], *c);
    }

    cout << "exp loss: " << ExpLoss(validation, validation_activations)
         << ", 0/1 loss: " << ZeroOneLoss(validation, validation_activations,
                                          &positive_loss, &negative_loss) << endl;
    cout << "+ err: " << positive_loss << ", - err: " << negative_loss << endl;

    cout << "To achieve + err of " << false_negative_rate << ": - err = " << fpr
         << ", bias = " << bias << endl;
    
    if (use_rates && (fpr < false_positive_rate)) {
      cout << "Desired false negative and false positive ( "
           << false_negative_rate << ", " << false_positive_rate
           << " ) acheived.  Stopping." << endl;
      break;
    }
  }
}

void TrainCascade(DataSource& data,
                  const vector<Feature>& features, int num_stages,
                  int max_positives, int max_negatives, Classifier *c) {
  //  vector<Patch> positive_patches;
  //  vector<Patch> negative_patches;
  vector<Patch> patches;
  //  vector<Patch> positive_validation_patches;
  //  vector<Patch> negative_validation_patches;
  vector<Patch> validation;

  c->type_ = Classifier::kCascade;
  c->filters_use_margin_ = false;
  c->filters_are_additive_ = false;
  c->filters_are_permanent_ = true;  

  cout << "Initial" << endl;

  for (int i = 0; i < num_stages; i++) {
    cout << endl << "Stage " << i << endl;
    cout <<         "-------------" << endl;

    patches.clear();
    validation.clear();

    Filter filt;
    if (i > 0) {
      filt.active_ = true;
      filt.threshold_ = c->chains_[i-1].biases_.back();
      filt.less_ = false;
    }
    c->chains_.push_back(Chain());
    c->filters_.push_back(filt);

    int num_positive = data.GetPositivePatchesActive(max_positives, *c, &patches);
    int num_negative = data.GetNegativePatchesActive(max_negatives, *c, &patches);

    int num_positive_validation = data.GetPositivePatchesActive(max_positives, *c, &validation);
    int num_negative_validation = data.GetNegativePatchesActive(max_negatives, *c, &validation);

    cout << "Loaded " << num_positive << " positive patches." << endl;
    cout << "Loaded " << num_negative << " negative patches." << endl;

    cout << patches.size() << endl;

    cout << "Loaded " << num_positive_validation << " positive validation patches." << endl;
    cout << "Loaded " << num_negative_validation << " negative validation patches." << endl;

    cout << validation.size() << endl;

    if (num_positive == 0 || num_negative == 0) {
      cout << "Unable to load positive or negative patches." << endl;
      return;
    }

    float false_negative_rate = 0.0;
    float false_positive_rate = 1.0;

    // for (unsigned int pp = 0; pp < positive_patches.size(); pp++) {
    //   if (Predict(positive_patches[pp], *c)) {
    //     patches.push_back(positive_patches[pp]);
    //   } else {
    //     false_negative_rate += 1.0;
    //   }
    // }
    // false_negative_rate = false_negative_rate / (float)positive_patches.size();
    
    // for (unsigned int np = 0; np < negative_patches.size(); np++) {  
    //   if (Predict(negative_patches[np], *c)) {
    //     patches.push_back(negative_patches[np]);
    //   } else {
    //     false_positive_rate += 1.0;
    //   }
    // }
    // false_positive_rate = 1.0 - (false_positive_rate / (float)negative_patches.size());

    cout << endl;
    cout << "False negative rate: " << false_negative_rate << endl;
    cout << "False positive rate: " << false_positive_rate << endl;
    cout << endl;

    if (false_positive_rate < 0.0001) {
      cout << "We're all done here..." << endl;
      break;
    }

    TrainStages(patches, vector<float>(), features, FLAGS_max_inner_stages, false, true,
		FLAGS_target_false_negative,
		FLAGS_target_false_positive_base - i*FLAGS_target_false_positive_step,
		validation, c);
    
    //c->stumps_.push_back(feat);
    //c->weights_.push_back(0.0);
    // Add a small bias just to make everything positive.
    //c->biases_.push_back(1e-8);
    //c->filters_.push_back(filt);
  }
}
  
void TrainBoosted(DataSource& data, const std::vector<Feature>& features,
		  int num_stages, int max_positives, int max_negatives, Classifier *c) {
  //  vector<Patch> positive_patches;
  //  vector<Patch> negative_patches;
  vector<Patch> patches;
  vector<float> sample_weights;

  vector<Patch> validation;

  if (FLAGS_anytime_boost) {
    c->type_ = Classifier::kAnytime;
    c->filters_use_margin_ = true;
    c->filters_are_additive_ = true;
    c->filters_are_permanent_ = false;
  }

  cout << "Initial" << endl;

  c->chains_.push_back(Chain());
  c->filters_.push_back(Filter());

  for (int i = 0; i < num_stages; i += FLAGS_stage_increment) {
    cout << endl << "Stage " << i << endl;
    cout <<         "-------------" << endl;

    patches.clear();
    sample_weights.clear();
    validation.clear();

    int num_positive = 0;
    int num_negative = 0;
    if (FLAGS_sample_patches) {
      data.GetPatchesSampled(max_positives + max_negatives, *c, &sample_weights, &patches);
      for (unsigned int p = 0; p < patches.size(); p++) {
      	if (patches[p].label() > 0)
      	  num_positive++;
      	else
      	  num_negative++;
      }
      //num_positive = data.GetPositivePatchesSampled(max_positives, *c, &sample_weights, &patches);
      //num_negative = data.GetNegativePatchesSampled(max_negatives, *c, &sample_weights, &patches);
    } else {
      num_positive = data.GetPositivePatches(max_positives, &patches);
      num_negative = data.GetNegativePatches(max_negatives, &patches);
    }
    cout << "Loaded " << num_positive << " positive patches." << endl;
    cout << "Loaded " << num_negative << " negative patches." << endl;
    if (num_positive == 0 || num_negative == 0) {
      cout << "Unable to load positive or negative patches." << endl;
      return;
    }

    int num_positive_validation = data.GetPositivePatches(max_positives, &validation);
    int num_negative_validation = data.GetNegativePatches(max_negatives, &validation);

    cout << "Loaded " << num_positive_validation << " positive validation patches." << endl;
    cout << "Loaded " << num_negative_validation << " negative validation patches." << endl;

    //    vector<Patch> patches;
    //    patches.insert(patches.end(), positive_patches.begin(), positive_patches.end());
    //    patches.insert(patches.end(), negative_patches.begin(), negative_patches.end());

    TrainStages(patches, sample_weights, features, FLAGS_stage_increment, true, false, 0.0, 0.0, validation, c);
  }

  if (FLAGS_anytime_boost) {
    c->filters_.pop_back();
    c->chains_.pop_back();
  }
}

void UpdateSingleStage(const vector<Patch>& patches, const Classifier& c,
                       int i, vector<float>* activations, vector<bool>* updated) {
  for (unsigned int p = 0; p < patches.size(); p++) {
    if (c.filters_are_permanent_ && !(*updated)[p]) {
      continue;
    }

    float v = (c.filters_use_margin_) ? abs((*activations)[p]) : (*activations)[p];
    if (c.filters_[i].PassesFilter(v)) {
      (*updated)[p] = true;
      if (c.filters_[i].active_ && !c.filters_are_additive_) {
	(*activations)[p] = 0.0;
      }

      for (unsigned int j = 0; j < c.chains_[i].stumps_.size(); j++) {
	float response = c.chains_[i].stumps_[j].Evaluate(patches[p]);
	(*activations)[p] += c.chains_[i].weights_[j] * response;
      }
    } else {
      (*updated)[p] = false;
    }
  }
}

void UpdateSingleStump(const vector<Patch>& patches, const Classifier& c,
                       int i, int j, vector<float>* activations, vector<bool>* updated) {
  for (unsigned int p = 0; p < patches.size(); p++) {
    if (c.filters_are_permanent_ && !(*updated)[p]) {
      continue;
    }

    if (j == 0) {
      float v = (c.filters_use_margin_) ? abs((*activations)[p]) : (*activations)[p];
      if (c.filters_[i].PassesFilter(v)) {
	(*updated)[p] = true;
	if (c.filters_[i].active_ && !c.filters_are_additive_) {
	  (*activations)[p] = 0.0;
	}
      } else {
	(*updated)[p] = false;
      }
    }

    if ((*updated)[p]) {
      float response = c.chains_[i].stumps_[j].Evaluate(patches[p]);
      (*activations)[p] += c.chains_[i].weights_[j] * response;
    }
  }
}

void OutputROC(string filename, const vector<Patch>& patches,
               const vector<float>& activations) {
  ofstream roc_file(filename.c_str(), ofstream::out);

  vector< pair<float, char> > sortable(patches.size());
  for (int p = 0; p < (int)(patches.size()); p++) {
    sortable[p].first = activations[p];
    sortable[p].second = patches[p].label();
  }

  sort(sortable.begin(), sortable.end());

  int positives = 0;
  int negatives = 0;

  for (int p = 0; p < (int)(patches.size() - 1); p++) {
    if (sortable[p].second > 0) {
      positives++;
    } else {
      negatives++;
    }
  }

  float true_positives = 0;
  float false_positives = 0;

  roc_file << "0,0" << endl;
  for (int p = (int)(patches.size() - 1); p >= 0; p--) {
    if (sortable[p].second > 0) {
      true_positives++;
    } else {
      false_positives++;
    }

    if (sortable[p].first == sortable[p + 1].first) continue;

    if ((p % 100) == 0) {
      roc_file << false_positives << "," << true_positives << endl;
    }
  }

  roc_file.close();
}

void GenerateStatistics(string filename, const vector<Patch>& patches,
			const Classifier& c, string roc_filename,
                        int roc_iteration) {
  vector<float> activations(patches.size(), 0.0);
  vector<bool> updated(patches.size(), true);

  ofstream stats_file(filename.c_str(), ofstream::out);

  float exp_loss, zero_one_loss;
  float positive_loss, negative_loss;
  float average_features = 0.0;
  exp_loss = ExpLoss(patches, activations);
  zero_one_loss = ZeroOneLoss(patches, activations,
                              &positive_loss, &negative_loss);
  cout << "Initial" << endl;
  cout << "exp loss: " << exp_loss
       << ", 0/1 loss: " << zero_one_loss << endl;
  cout << "+ err: " << positive_loss << ", - err: " << negative_loss << endl;
  
  stats_file << "iteration,exploss,error,pos_error,neg_error,threshold,updated,avgfeat" << endl;
  stats_file << 0 << "," << exp_loss << "," << zero_one_loss << ","
             << positive_loss << "," << negative_loss << ","
             << 0 << "," << 0 << "," << 1.0 << "," << average_features << endl;

  for (int i = 0; i < (int)(c.chains_.size()); i++) {
    cout << endl << "Stage " << i << endl;
    cout <<         "-------------" << endl;
    
    cout << endl << "Selected filter:" << endl;
    c.filters_[i].Print();

    cout << endl << "Selected features:" << endl;
    for (unsigned int j = 0; j < c.chains_[i].stumps_.size(); j++) { 
      UpdateSingleStump(patches, c, i, j, &activations, &updated);

      cout << endl << "*** Stump " << j << " ***" << endl;
      c.chains_[i].stumps_[j].Print();

      cout << "alpha: " << c.chains_[i].weights_[j] << endl;

      exp_loss = ExpLoss(patches, activations);
      zero_one_loss = ZeroOneLoss(patches, activations,
				  &positive_loss, &negative_loss);
      cout << "exp loss: " << exp_loss
	   << ", 0/1 loss: " << zero_one_loss << endl;
      cout << "+ err: " << positive_loss << ", - err: " << negative_loss << endl;
    
      int update_count = 0;
      for (unsigned int p = 0; p < updated.size(); p++) {
	if (updated[p])
	  update_count++;
      }
      
      average_features = average_features + ((float)c.chains_[i].stumps_.size()) * (update_count / (float)updated.size());
      stats_file << i << "," << exp_loss << "," << zero_one_loss << ","
		 << positive_loss << "," << negative_loss << ","
		 << c.filters_[i].threshold_ << ","
		 << update_count / (float)(updated.size()) << "," << average_features << endl;
      
    }

    if ((roc_filename != "") && (i == roc_iteration)) {
      OutputROC(roc_filename, patches, activations);
    }

    
    // float false_positive_rate;
    // float bias = ComputePredictionBias(patches, activations, false, &false_positive_rate);
    // cout << "To achieve + err of " << false_negative_rate << ": - err = " << false_positive_rate 
    //      << ", bias = " << bias << endl;
  }
  
  stats_file.close();
}

}  // namespace speedboost
