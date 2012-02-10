//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <iostream>
#include <omp.h>
#include <string>

#include "classifier.h"
#include "data_source.h"
#include "feature.h"
#include "patch.h"

using namespace speedboost;
using namespace std;

DEFINE_string(features_filename, "random.features",
              "File to save or load generated features from.");
DEFINE_string(positive_patches_glob, "",
              "File glob containing saved positive patches.");
DEFINE_string(negative_patches_glob, "",
              "File glob containing saved negative patches.");
DEFINE_string(classifier_filename, "scratch.classifier",
              "File to save the trained classifier to.");
DEFINE_int32(num_features, 16000,
             "Number of features to generate.");
DEFINE_int32(max_negatives, 50000, "Maximum number of negatives");
DEFINE_int32(max_positives, 10000, "Maximum number of positives");
DEFINE_int32(num_stages, 100, "Number of stages to train in the classifier");
DEFINE_bool(cascade, false, "Build a cascade.");
DEFINE_int32(omp_num_threads, 10, "Number of openmp threads to use for feature selection.");

bool ValidateInputFilename(const char* flagname, const string& value) {
  ifstream in(value.c_str(), ifstream::in);
  if (in.is_open()) {
    in.close();
    return true;
  }

  in.close();
  printf("File does not exist for flag --%s: \n  \"%s\"\n", flagname, value.c_str());

  return false;
}

//static const bool ppf_dummy = google::RegisterFlagValidator(&FLAGS_positive_patches_glob, &ValidateInputFilename);
//static const bool pnf_dummy = google::RegisterFlagValidator(&FLAGS_negative_patches_glob, &ValidateInputFilename);

int main(int argc, char* argv[])
{
  // parse up the flags
  google::ParseCommandLineFlags(&argc, &argv, true);

  omp_set_num_threads(FLAGS_omp_num_threads);

  DataSource data(FLAGS_positive_patches_glob, FLAGS_negative_patches_glob);

  vector<Feature> features;
  if (!Feature::ReadFeaturesFromFile(FLAGS_features_filename, &features)) {
    Feature::GenerateFeatures(FLAGS_num_features, &features);
    Feature::WriteFeaturesToFile(FLAGS_features_filename, features);
    cout << "Generated and saved " << FLAGS_num_features << " features." << endl;
  } else {
    cout << "Loaded " << FLAGS_num_features << " features." << endl;
  }

  Classifier c;
  if (FLAGS_cascade) {
    TrainCascade(data, features, FLAGS_num_stages,
		 FLAGS_max_positives, FLAGS_max_negatives, &c);
  //                FLAGS_positive_validation_filename, FLAGS_negative_validation_filename,
  } else {
    TrainBoosted(data, features, FLAGS_num_stages,
		 FLAGS_max_positives, FLAGS_max_negatives, &c);
  }

  cout << "Learned classifier:" << endl << endl;
  c.Print();

  c.WriteToFile(FLAGS_classifier_filename);

  return 0;
}
