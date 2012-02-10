//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <iostream>
#include <string>

#include "classifier.h"
#include "feature.h"
#include "patch.h"

using namespace speedboost;
using namespace std;

DEFINE_string(positive_patches_glob, "",
              "File glob containing saved positive patches.");
DEFINE_string(negative_patches_glob, "",
              "File glob containing saved negative patches.");
DEFINE_string(classifier_filename, "scratch.classifier",
              "File to save the trained classifier to.");
DEFINE_string(statistics_filename, "stats.csv",
              "File to save prediction statistics to.");
DEFINE_int32(max_negatives, 50000, "Number of negative samples to use");
DEFINE_int32(max_positives, 10000, "Number of positive samples to use");
DEFINE_int32(roc_output_iteration, 100, "");
DEFINE_string(roc_output, "results/roc.csv", "");

int main(int argc, char*argv[])
{
  // parse up the flags
  google::ParseCommandLineFlags(&argc, &argv, true);

  DataSource data(FLAGS_positive_patches_glob, FLAGS_negative_patches_glob);
  vector<Patch> patches;
  int num_positive = data.GetPositivePatches(FLAGS_max_positives, &patches);
  int num_negative = data.GetNegativePatches(FLAGS_max_negatives, &patches);
  if (num_positive == 0) {
    cout << "Failed to load positive patches." << endl;
    return 1;
  }
  if (num_negative == 0) {
    cout << "Failed to load positive patches." << endl;
    return 1;
  }

  Classifier c;
  c.ReadFromFile(FLAGS_classifier_filename);

  GenerateStatistics(FLAGS_statistics_filename, patches, c,
                     FLAGS_roc_output, FLAGS_roc_output_iteration);

  return 0;
}
