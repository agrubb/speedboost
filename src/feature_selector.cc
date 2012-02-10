//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "feature.h"
#include "feature_selector.h"
#include "patch.h"

using namespace std;

DEFINE_int32(threshold_min_examples, 500, "Minimum number of examples per each threshold section.");
DEFINE_int32(threshold_examples_step, 100, "Minimum examples between each threshold section.");
DEFINE_int32(threshold_min_positive_examples, 50, "Minimum positive examples per each threshold section.");
DEFINE_int32(threshold_min_negative_examples, 50, "Minimum negative examples per each threshold section.");
DEFINE_double(threshold_min_delta, 0.01, "Minimum change in threshold per section.");

namespace speedboost {

FeatureSelector::FeatureSelector(const vector<Patch>& patches, const vector<Feature>& feats)
  : labels(patches.size()),
    responses(feats.size()),
    sorted(feats.size()),
    features(&feats)
{
  for (unsigned int p = 0; p < patches.size(); p++) {
    labels[p] = patches[p].label();
  }
  
  #pragma omp parallel default(shared)
  {
    vector< pair<float, int> > sortable(patches.size());
    #pragma omp for
    for (unsigned int f = 0; f < features->size(); f++) {
      responses[f].resize(patches.size());
      sorted[f].resize(patches.size());
      
      for (unsigned int p = 0; p < patches.size(); p++) {
        responses[f][p] = (*features)[f].Evaluate(patches[p]);
        sortable[p].first = (*features)[f].Evaluate(patches[p]);
        sortable[p].second = p;
      }
      
      sort(sortable.begin(), sortable.end());
      for (unsigned int p = 0; p < patches.size(); p++) {
        sorted[f][p] = sortable[p].second;
      }
    }
  }
}

void FeatureSelector::SelectFeatureSingle(const vector<float>& weights, const vector<float>& activations,
                                          int index, float positive_weight, float negative_weight,
                                          float* split, float* sign, float* loss)
{
  float positive_weight_below = 0.0;
  float negative_weight_below = 0.0;
  float positive_weight_above = positive_weight;
  float negative_weight_above = negative_weight;

  float best_split = FLT_MIN;
  float best_sign = (positive_weight_above > negative_weight_above) ? 1 : -1;
  float best_loss = min(positive_weight_above, negative_weight_above);

  for (unsigned int i = 1; i < responses[index].size(); i++) {
    int p1 = sorted[index][i - 1];
    int p2 = sorted[index][i];

    if (labels[p1] > 0) {
      positive_weight_above -= weights[p1];
      positive_weight_below += weights[p1];
    } else {
      negative_weight_above -= weights[p1];
      negative_weight_below += weights[p1];
    }

    if (responses[index][p1] == responses[index][p2]) continue;

    float positive_loss = negative_weight_above + positive_weight_below;
    float negative_loss = positive_weight_above + negative_weight_below;
    
    if (best_loss > min(positive_loss, negative_loss)) {
      best_loss = min(positive_loss, negative_loss);
      best_sign = (positive_loss < negative_loss) ? 1 : -1;
      best_split = (responses[index][p1] + responses[index][p2]) / 2.0;
    }
  }

  *loss = best_loss;
  *split = best_split;
  *sign = best_sign;  
}

void FeatureSelector::SelectFeatureBucketedSingle(const vector<float>& weights, const vector<float>& activations,
                                                  int index, const vector<int>& buckets,
                                                  const vector<float>& positive_weight, const vector<float>& negative_weight,
                                                  const vector<float>& loss, const vector<float>& tau,
                                                  float* split, float* sign, float* err, float *gain, int* bucket)
{
  int num_buckets = positive_weight.size();
  vector<float> positive_weight_below(num_buckets, 0.0);
  vector<float> negative_weight_below(num_buckets, 0.0);

  vector<int> best_index(num_buckets, 0);
  vector<float> best_sign(num_buckets, 1.0);
  vector<float> best_inner_product(num_buckets, 0.0);

  for (int b = 0; b < num_buckets; b++) {
    best_inner_product[b] = (positive_weight[b] > negative_weight[b]) ?
      (positive_weight[b] - negative_weight[b]) : (negative_weight[b] - positive_weight[b]);
  }
  
  // cout << "Finding best split + thresh for feature " << index << endl;
  //  float best_split = FLT_MIN;
  //  float best_sign = (positive_weight[num_buckets - 1] > negative_weight[num_buckets - 1]) ? 1 : -1;
  //  float best_gain = FLT_MIN;
  //  float best_loss = min(positive_weight[num_buckets - 1], negative_weight[num_buckets - 1]) / 
  //    (positive_weight[num_buckets - 1] + negative_weight[num_buckets - 1]);
  //  float best_threshold = max_threshold;

  for (unsigned int i = 1; i < responses[index].size(); i++) {
    int p1 = sorted[index][i - 1];
    int p2 = sorted[index][i];

    if (labels[p1] > 0) {
      for (int b = buckets[p1]; b < num_buckets; b++) {
        positive_weight_below[b] += weights[p1];
      }
    } else {
      for (int b = buckets[p1]; b < num_buckets; b++) {
        negative_weight_below[b] += weights[p1];
      }
    }

    if (responses[index][p1] == responses[index][p2]) continue;

    // cout << "example " << i << " bucket: " << bucket << ", activation: " << activations[p1]
    //      << ", label: " << (int)labels[p1] << ", weight: " << weights[p1] << endl;

    for (int b = buckets[p1]; b < num_buckets; b++) {
      // pa + nb - (na + pb) = p - pb + nb - (n - nb + pb) = p - 2 pb - (n - 2 nb)

      float positive_diff = positive_weight[b] - 2 * positive_weight_below[b];
      float negative_diff = negative_weight[b] - 2 * negative_weight_below[b];
      float positive_inner_product = (positive_diff - negative_diff);
      float negative_inner_product = (negative_diff - positive_diff);

      if (best_inner_product[b] < max(positive_inner_product, negative_inner_product)) {
        best_inner_product[b] = max(positive_inner_product, negative_inner_product);
        best_index[b] = i;
        best_sign[b] = (positive_inner_product > negative_inner_product) ? 1 : -1;
      }
    }
  }

  int best_bucket = 0;
  float best_gain = FLT_MIN;
  
  //cout << "Buckets were:" << endl;
  for (int b = 0; b < num_buckets; b++) {
    float inner_product = best_inner_product[b] / (positive_weight[b] + negative_weight[b]);
    // float frac = (1 - sqrt(1 - inner_product*inner_product));
    float delta_loss = loss[b] * (1 - sqrt(1 - inner_product*inner_product));
    float gain = delta_loss / tau[b];

    //cout << "ip: " << inner_product << " frac: " << frac
    //	 << " dL: " << delta_loss << " gain: " << gain << " index: " << best_index[b] << endl;
    //cout << "Bucket is: " << b << endl;

    if (best_gain < gain) {
      best_gain = gain;
      best_bucket = b;
    }
  }

  //  cout << "best bucket: " << best_bucket << endl;

  int i = best_index[best_bucket];
  float sum;
  if (i == 0) {
    sum = FLT_MIN;
  } else {
    sum = responses[index][sorted[index][i - 1]];
    while ((i < num_buckets) && (buckets[sorted[index][i]] > best_bucket)) {
      i++;
    }
    sum += responses[index][sorted[index][i]];
  }

  *gain = best_gain;
  *err = 0.5 - 0.5 * best_inner_product[best_bucket] /
    (positive_weight[best_bucket] + negative_weight[best_bucket]);
  *split = sum / 2.0;
  *sign = best_sign[best_bucket];
  *bucket = best_bucket;
}

void FeatureSelector::UpdateActivations(const DecisionStump& feature, const Filter& filter,
                                        int index, float alpha, vector<float>* activations)
{
  cout << "Updating activations: " << endl;
  cout << "index: " << index << " alpha: " << alpha << endl;
  for (unsigned int i = 0; i < activations->size(); i++) {
    float response = responses[index][i];
    
    if (filter.PassesFilter(abs((*activations)[i]))) {
      (*activations)[i] += alpha * feature.Evaluate(response);
    }
  }
}

DecisionStump FeatureSelector::SelectFeature(const vector<float>& weights, const vector<float>& activations,
                                              int *index, float* err)
{
  float positive_weight = 0.0;
  float negative_weight = 0.0;

  for (unsigned int i = 0; i < weights.size(); i++) {
    if (labels[i] > 0) {
      positive_weight += weights[i];
    } else {
      negative_weight += weights[i];
    }
  }

  float best_loss = FLT_MAX;
  float best_feature = -1;
  float best_split = 0.0;
  float best_sign = 1.0;

  vector<float> losses(features->size());
  vector<float> splits(features->size());
  vector<float> signs(features->size());

  #pragma omp parallel for
  for (unsigned int i = 0; i < features->size(); i++) {
    float loss;
    float split;
    float sign;
    
    SelectFeatureSingle(weights, activations, i, positive_weight, negative_weight,
                        &split, &sign, &loss);
    losses[i] = loss;
    splits[i] = split;
    signs[i] = sign;
  }

  for (unsigned int i = 0; i < features->size(); i++) {
    if (losses[i] < best_loss) {
      best_loss = losses[i];
      best_feature = i;
      best_split = splits[i];
      best_sign = signs[i];
    }
  }

  DecisionStump lf((*features)[best_feature],
		   best_split,
		   best_sign);

  *index = best_feature;
  *err = best_loss / (positive_weight + negative_weight);

  return lf;
}

int Bucket(float activation, float min_threshold, float max_threshold, int num_buckets)
{
  float fraction = (abs(activation) - min_threshold) / (max_threshold - min_threshold);
  
  return max(0, (int)floor(fraction * (num_buckets - 1) + 1));
}

void BuildBuckets(const vector<char>& labels, const vector<float>& weights,
                  const vector<float>& activations,
                  vector<int>* buckets, vector<float>* thresholds)
{
  // float min_threshold = 0.01;
  // float max_threshold = 1.0;
  // float num_buckets = 20;

  // thresholds->resize(num_buckets);
  // for (int b = 0; b < num_buckets; b++) {
  //   (*thresholds)[b] = min_threshold + b * (max_threshold - min_threshold) / ((float)(num_buckets - 1));
  // }

  // for (int i = 0; i < weights.size(); i++) {
  //   (*buckets)[i] = Bucket(activations[i], min_threshold, max_threshold, num_buckets);
  // }

  vector< pair<float, int> > sortable(activations.size());
  for (unsigned int p = 0; p < activations.size(); p++) {
    sortable[p].first = abs(activations[p]);
    sortable[p].second = p;
  }

  sort(sortable.begin(), sortable.end());

  float last = 0.0;
  int count = 0;
  int positive_count = 0;
  int negative_count = 0;
  int bucket = 0;
  for (int p = 1; p < (int)(activations.size()); p++) {
    int p1 = sortable[p - 1].second;
    // int p2 = sortable[p].second;

    if (labels[p1] > 0)
      positive_count++;
    else
      negative_count++;

    count++;
    (*buckets)[p1] = bucket;

    if (sortable[p - 1].first == sortable[p].first) continue;
    if (p < FLAGS_threshold_min_examples) continue;
    if (count < FLAGS_threshold_examples_step) continue;
    if (positive_count < FLAGS_threshold_min_positive_examples) continue;
    if (negative_count < FLAGS_threshold_min_negative_examples) continue;

    float threshold = (sortable[p - 1].first + sortable[p].first) / 2.0;
    
    if ((threshold - last) < FLAGS_threshold_min_delta) continue;
  
    thresholds->push_back(threshold);
    last = threshold;
    positive_count = 0;
    negative_count = 0;
    count = 0;
    bucket++;
  }

  (*buckets)[buckets->size() - 1] = bucket;
  thresholds->push_back(FLT_MAX);

  cout << "Thresholds: ";
  for (unsigned int b = 0; b < thresholds->size(); b++) {
    cout << (*thresholds)[b] << " ";
  }
  cout << endl;
}

void BucketedLosses(const vector<char>& labels, const vector<float>& activations,
                    const vector<int>& buckets, vector<float>* loss) {  
  for (unsigned int b = 0; b < loss->size(); b++) {
    (*loss)[b] = 0.0;
  }

  for (unsigned int i = 0; i < labels.size(); i++) {
    float y = (labels[i] > 0) ? 1.0 : -1.0;
    for (unsigned int b = buckets[i]; b < loss->size(); b++) {
      (*loss)[b] += exp(- y * activations[i]);
    }
  }
}

DecisionStump FeatureSelector::SelectFeatureAndThreshold(const vector<float>& weights, const vector<float>& activations,
                                                          int *ret_index, float* ret_error, float* ret_threshold) {
  float min_act = FLT_MAX;
  float max_act = FLT_MIN;
  float min_abs_act = FLT_MAX;
  float max_abs_act = 0.0;

  cout << endl << "Selecting feature..." << endl;

  for (unsigned int i = 0; i < activations.size(); i++) {
    min_act = min(min_act, activations[i]);
    max_act = max(max_act, activations[i]);
    min_abs_act = min(min_abs_act, abs(activations[i]));
    max_abs_act = max(max_abs_act, abs(activations[i]));
  }

  cout << "act = [" << min_act << ", " << max_act << "], "
       << "|act| = [" << min_abs_act << ", " << max_abs_act << "], " << endl;
  
  vector<int> buckets(weights.size());
  vector<float> thresholds;

  BuildBuckets(labels, weights, activations, &buckets, &thresholds);
  int num_buckets = thresholds.size();

  vector<float> positive_weight(num_buckets);
  vector<float> negative_weight(num_buckets);
  vector<float> tau(num_buckets);
  vector<float> loss(num_buckets);

  for (unsigned int i = 0; i < weights.size(); i++) {
    if (labels[i] > 0) {
      for (int b = buckets[i]; b < num_buckets; b++) {
        positive_weight[b] += weights[i];
        tau[b] += 1;
      }
    } else {
      for (int b = buckets[i]; b < num_buckets; b++) {
        negative_weight[b] += weights[i];
        tau[b] += 1;
      }
    }
  }

  for (int b = 0; b < num_buckets; b++) {
    tau[b] /= ((float)activations.size());
  }

  cout << "Taus: ";
  for (unsigned int b = 0; b < tau.size(); b++) {
    cout << tau[b] << " ";
  }
  cout << endl;

  BucketedLosses(labels, activations,
                 buckets, &loss);

  // cout << "Weight totals:" << endl;
  //  for (int b = 0; b < num_buckets; b++) {
  //    cout << positive_weight[b] << " " << negative_weight[b] << endl;
  //  }

  float best_error = 0.5;
  float best_gain = FLT_MIN;
  float best_feature = -1;
  float best_split = 0.0;
  float best_sign = 1.0;
  int best_bucket = 0;

  vector<float> errors(features->size());
  vector<float> gains(features->size());
  vector<float> splits(features->size());
  vector<float> signs(features->size());
  vector<int> best_buckets(features->size());

  #pragma omp parallel for
  for (unsigned int i = 0; i < features->size(); i++) {
    float error;
    float gain;
    float split;
    float sign;
    int bucket;
    
    SelectFeatureBucketedSingle(weights, activations, i, buckets,
                                positive_weight, negative_weight, loss, tau,
                                &split, &sign, &error, &gain, &bucket);
    errors[i] = error;
    gains[i] = gain;
    splits[i] = split;
    signs[i] = sign;
    best_buckets[i] = bucket;
  }

  for (unsigned int i = 0; i < features->size(); i++) {
    if (gains[i] > best_gain) {
      best_gain = gains[i];
      best_error = errors[i];
      best_feature = i;
      best_split = splits[i];
      best_sign = signs[i];
      best_bucket = best_buckets[i];
    }
  }

  DecisionStump lf((*features)[best_feature],
		   best_split,
		   best_sign);

  *ret_index = best_feature;
  *ret_error = best_error;
  *ret_threshold = thresholds[best_bucket];

  return lf;
}

}  // namespace speedboost
