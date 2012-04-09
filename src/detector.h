//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_DETECTOR_H
#define SPEEDBOOST_DETECTOR_H

#include <stdio.h>
#include <sys/time.h>

#include "classifier.h"
#include "patch.h"
#include "feature.h"

namespace speedboost {

class Detector;

/**
 * Sequencer is used to figure out the next point an example will be updated at.
 * Currently used only for SpeedBoost trained classifiers.
 */
class Sequencer {
public:
  Sequencer(Classifier* c);
  
  /**
   * Given a current chain to start searching at,
   * find the next chain an example with activation
   * will be updated at.
   */
  int NextChain(int current_chain, float activation) const;
  
private:
  Classifier* c_;
  
  std::vector<int> next_biggest_;
  std::vector<float> max_threshold_;
};

/**  
 * Run the anytime detection code for a single scale of the
 * activation pyramid.
 */
class SingleScaleDetector {
public:
  /**
   * Construct a SingleScaleDetector with a classifier and the scaled integral image.
   * These objects should not be freed while SingleScaleDetector is still
   * being used.
   */
  SingleScaleDetector(Classifier* c, Patch* integral);

  /**
   * Functions to evaluate a decision stump for every patch in the scaled image.
   */
  void EvaluateAllPatches(float weight, const DecisionStump& stump, const Patch& frame, Patch* activations);
  /**
   * Filter the patches using current activations and filtering criteria in filter.
   */
  void EvaluateAllPatchesFiltered(float weight, const DecisionStump& stump, const Patch& frame,
                                  const Filter& filter, Patch* activations);
  /**
   * Use a list of indices into the frame to evaluate just those patch locations.
   */
  void EvaluateAllPatchesListed(float weight, const DecisionStump& stump, const Patch& frame,
                                const std::vector<int>& indices, Patch* activations);

  /**
   * True if there are features (decision stumps) that can still be evaluated
   * on the frame.
   */
  bool HasMoreFeatures();
  /**
   * Update the activations using the next feature in the classifier.
   * If updates is non-null, the updates patch is used to store the number
   * of times each pixel locations is updated.
   */
  void ComputeNextFeature(const Sequencer& seq, Patch* activations,
			  Patch* updates = NULL);
  
  /**
   * Return the average number of features per pixel this detector
   * has computed so far.
   */
  float FeaturesPerPixel() { return (float)updated_pixels_ / (float)num_pixels_; }

private:
  Classifier* c_;
  Patch* integral_;

  int chain_index_;
  int stump_index_;

  std::vector<int> default_indices_;
  std::vector< std::vector<int> > indices_;

  int num_pixels_;
  int updated_pixels_;
};

/**
 * Detector for running actual detection on an image.
 */
class Detector {
public:
  Detector(Classifier* c);

  /**
   * Builds and computes the activation pyramid for a given frame.
   * After computing the pyramid will contain num_scales patches,
   * with each one shrinking sucessively by scaling_factor.
   * This shrinkage corresponds to a scaling_factor increase
   * in the patch size.
   *
   * Each activation frame will be the same size as the scaled image
   * for that level of the pyramid.  Each pixel in the activation
   * frame corresponds to a detection of size [patch_width x patch_height]
   * with the upper left corner at the pixel location in the activation
   * frame.  This means there will be a strip on the lower and right edges
   * with 0 activation, as no patches can start at these pixels.
   */
  void ComputeActivationPyramid(const Patch& frame, int num_scales, float scaling_factor,
                                std::vector<Patch>* activation_pyramid);

  /**
   * Compute the activation pyramid, and then rescale each level back to
   * the original image size and merge the results by taking the maxmimum
   * activation for each pixel across all scales.
   */
  void ComputeMergedActivation(const Patch& frame, int num_scales, float scaling_factor,
                               Patch* merged);

  /**
   * Compute the detections for the frame corresponding to the activations
   * in the activation pyramid.
   */
  void ComputeDetections(const Patch& frame, int num_scales, float scaling_factor, float detection_threshold,
                         std::vector<Label>* detections);

  /**
   * Filter out the overlapping detections.
   */
  void FilterDetections(const std::vector<Label>& detections, const std::vector<float>& weights,
                        float overlap, std::vector<Label>* filtered);

  void Tic() {
    gettimeofday(&start_, NULL);
  }

  double Toc() {
    struct timeval elapsed;
    gettimeofday(&end_, NULL);
    timersub(&end_, &start_, &elapsed);
    return double(elapsed.tv_sec + (double)(elapsed.tv_usec) / 1e6);
  }

protected:
  void SetupForFrame(const Patch& frame, int num_scales, float scaling_factor,
                     std::vector<Patch>* scaled_integrals, std::vector<Patch>* scaled_activations,
                     std::vector<SingleScaleDetector>* scaled_detectors);

  Classifier* c_;
  Sequencer sequencer_;

  struct timeval start_, end_;
};

}  // namespace speedboost

#endif  // ifndef SPEEDBOOST_DETECTOR_H
