//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_FEATURE_SELECTOR_H
#define SPEEDBOOST_FEATURE_SELECTOR_H

#include <stdio.h>

#include "classifier.h"
#include "feature.h"
#include "patch.h"

namespace speedboost {

class FeatureSelector {
public:
  FeatureSelector(const std::vector<Patch>& patches, const std::vector<Feature>& features);

  void SelectFeatureSingle(const std::vector<float>& weights, const std::vector<float>& activations,
                           int index, float positive_weight, float negative_weight,
                           float* split, float* sign, float* loss);
  void SelectFeatureBucketedSingle(const std::vector<float>& weights, const std::vector<float>& activations,
                                   int index, const std::vector<int>& buckets,
                                   const std::vector<float>& positive_weight, const std::vector<float>& negative_weight,
                                   const std::vector<float>& loss, const std::vector<float>& tau,
                                   float* split, float* sign, float* err, float *gain, int* bucket);
  DecisionStump SelectFeature(const std::vector<float>& weights, const std::vector<float>& activations,
                               int* index, float* err);
  DecisionStump SelectFeatureAndThreshold(const std::vector<float>& weights, const std::vector<float>& activations,
                                           int* index, float* err, float* threshold);
  
  void UpdateActivations(const DecisionStump& feature, const Filter& filter,
                         int index, float alpha, std::vector<float>* activations);
  
  std::vector<char> labels;
  std::vector< std::vector<float> > responses;
  std::vector< std::vector<int> > sorted;

  const std::vector<Feature>* features;
};

}  // namespace speedboost

#endif  // ifndef SPEEDBOOST_FEATURE_SELECTOR_H
