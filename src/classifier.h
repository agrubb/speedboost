//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_CLASSIFIER_H
#define SPEEDBOOST_CLASSIFIER_H

#include <stdio.h>

#include "classifier.pb.h"
#include "data_source.h"
#include "feature.h"
#include "patch.h"

DECLARE_bool(anytime_boost);

namespace speedboost {

/**
 * Decision stump over the outputs of the Feature class.
 * Underlying feature is given by base.
 * If base feature outputs v then this stump
 * will output -sign if v < split and
 * sign if v >= split.
 */
class DecisionStump {
public:
  DecisionStump() {}

  DecisionStump(const Feature& base, float split, float sign)
    : base_(base), split_(split), sign_(sign) {}

  // bool operator==(const DecisionStump& other) const {
  //   return (base_ == other.base_ &&
  //           split_ == other.split_ &&
  //           sign_ == other.sign_);
  // }

  void Print() const {
    base_.Print();
    printf("  < %f == %f\n", split_, -sign_);
  }

  /**
   * Evaluate either on a patch or on the float value
   * output by a Feature object for some patch.
   */
  float Evaluate(const Patch& p) const;
  float Evaluate(float response) const;

  /**
   * Conversion to and from protobuf objects.
   */
  bool FromMessage(const StumpMessage& msg);
  void ToMessage(StumpMessage* msg) const;

  /**
   * File input and output.
   */
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;

  Feature base_;
  float split_;
  float sign_;
};

/**
 * Represents a filter for selecting which examples
 * to run a set of decision stumps on.
 *
 * Only filters data if active == true.
 */
class Filter {
public:
  Filter()
    : active_(false), threshold_(0), less_(true) {}

  void Print() const {
    if (active_) {
      if (less_) {
        printf("Filter: < %f\n", threshold_);
      } else {
        printf("Filter: > %f\n", threshold_);
      }
    } else {
      printf("Filter: INACTIVE.\n");
    }
  }

  /**
   * If PassesFilter returns true, a patch with
   * activation will be updated.  If false,
   * the patch will not be updated.
   */
  bool PassesFilter(float activation) const {
    if (active_) {
      if (less_) {
        return activation < threshold_;
      } else {
        return activation > threshold_;
      }
    }

    return true;
  }

  /**
   * Conversion to and from protobuf objects.
   */
  bool FromMessage(const FilterMessage& msg);
  void ToMessage(FilterMessage* msg) const;

  /**
   * File input and output.
   */
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;

  bool active_;
  float threshold_;
  bool less_;
};


/**
 * A sequence of weighted decision stumps.
 */
class Chain {
public:
  Chain()
    : stumps_(),
      weights_(),
      biases_()
  {}

  /**
   * Conversion to and from protobuf objects.
   */
  bool FromMessage(const ChainMessage& msg);
  void ToMessage(ChainMessage* msg) const;

  /**
   * File input and output.
   */
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;

  std::vector<DecisionStump> stumps_;
  std::vector<float> weights_;
  std::vector<float> biases_;
};

/**
 * Classifier object.
 * Store a sequence of chains (see above),
 * along with a filter that corresponds to each chain.
 * If a patch passes the corresponding filter, it will
 * be updated with the output for that chain.
 *
 * Some other options are described below.
 */
class Classifier {
public:
  enum ClassifierType {
    kBoosted = 0,
    kCascade,
    kAnytime,
  };

  Classifier()
    : type_(kBoosted),
      filters_use_margin_(false),
      filters_are_additive_(false),
      filters_are_permanent_(false) {}

  void Print() const;

  /**
   * Conversion to and from protobuf objects.
   */
  bool FromMessage(const ClassifierMessage& msg);
  void ToMessage(ClassifierMessage* msg) const;

  /**
   * File input and output.
   */
  bool Read(std::istream& in);
  void Write(std::ostream& out) const;

  /**
   * Read or write the classifier to a file as
   * human readable text.
   */
  bool ReadFromFile(const std::string& filename);
  bool WriteToFile(const std::string& filename) const;

  /**
   * Returns true if patch p passes the filter at the
   * final chain of this classifier.
   */
  bool IsActiveInLastChain(const Patch& p) const;

  /**
   * Return the activation for patch p using this
   * classifier.
   */
  float Activation(const Patch &p) const;

  ClassifierType type_;

  std::vector<Chain> chains_;
  std::vector<Filter> filters_;

  // Whether the filters use the margin or the actual activation.
  // E.g. use |f(x)| < c or f(x) < c
  bool filters_use_margin_;

  // If filters are not additive, when an example passes a filter
  // the activation for that example will be reset to 0, as in cascades.
  // Otherwise, the old activation is maintained and added too,
  // as in regular boosting and SpeedBoost.
  bool filters_are_additive_;

  // If filters are permanent, once an example does not pass a filter,
  // it is never updated again, as in cascades.
  // If not permanent, only the corresponding chain is skipped,
  // and the next filter is then run, as in SpeedBoost.
  bool filters_are_permanent_;
};

/**
 * Update activations using stump j from chain i in classifier c.
 * Which patches are / have been updated are tracked in the updated vector.
 * Both activations and updated should be the same size as patches.
 */
void UpdateSingleStump(const std::vector<Patch>& patches, const Classifier& c,
                       int i, int j, std::vector<float>* activations, std::vector<bool>* updated);

/**
 * Train a cascade on max_positives and max_negatives training patches
 * from data, with at most num_stages stages.
 */
void TrainCascade(DataSource& data, const std::vector<Feature>& features,
		  int num_stages, int max_positives, int max_negatives, Classifier* c);

/**
 * Train a boosted (or possible SpeedBoost) classifier on max_positives
 * and max_negatives training patches from data,
 * with at most num_stages stages.
 */
void TrainBoosted(DataSource& data, const std::vector<Feature>& features,
		  int num_stages, int max_positives, int max_negatives, Classifier* c);

/**
 * Generate prediction statistics for given patches using classifier c.
 * Outputs a csv file containing the statistics to filename,
 * and optionally an ROC curve for a given iteration (number of features evaluated)
 * to roc_filename.
 */
void GenerateStatistics(std::string filename, const std::vector<Patch>& patches,
			const Classifier& c, std::string roc_filename,
                        int roc_iteration);

}  // namespace speedboost

#endif  // ifndef SPEEDBOOST_CLASSIFIER_H
