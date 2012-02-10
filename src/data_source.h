//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_DATA_SOURCE_H
#define SPEEDBOOST_DATA_SOURCE_H

#include <gflags/gflags.h>
#include <fstream>
#include <string>
#include <vector>

#include "patch.h"

namespace speedboost {

class Classifier;

/**
 * Object for reading and sampling training data, etc. from
 * multiple files on disk.  Data should have already been
 * converted to the binary format (see load.cc).
 */
class DataSource {
public:
  /**
   * Give a set of positive files and negative files containing patches
   * for streaming as positive and negative data.
   */
  DataSource(const std::string& positive_file_glob, const std::string& negative_file_glob);

  /**
   * Use a single set of files containing the entire set of training patches,
   * coupled with labels indicating where the positive samples are in the data.
   * Eventually will be used to automatically extract negative data from the
   * areas that aren't explicitly labeled positive.
   *
   * NOTE: Not completely implemented yet.
   */
  DataSource(const std::string& frames_file_glob);

  /**
   * Just get the next max_num_patches patches from the data stream.
   */
  int GetPositivePatches(int max_num_patches, std::vector<Patch>* patches);
  int GetNegativePatches(int max_num_patches, std::vector<Patch>* patches);
  
  /**
   * As above, but filter out the patches so only patches that are 'active' in the last
   * chain of classifier c are returned.  Used for getting data for cascade training.
   */
  int GetPositivePatchesActive(int max_num_patches, const Classifier& c, std::vector<Patch>* patches);
  int GetNegativePatchesActive(int max_num_patches, const Classifier& c, std::vector<Patch>* patches);

  /**
   * Get max_num_patches patches from the data stream, where the patches are
   * sampled according to their current activation using classifier c.
   * The weights vector represents the weights used to adjust for sampling.
   */
  int GetPositivePatchesSampled(int max_num_patches, const Classifier& c,
        			std::vector<float>* weights, std::vector<Patch>* patches);
  int GetNegativePatchesSampled(int max_num_patches, const Classifier& c,
        			std::vector<float>* weights, std::vector<Patch>* patches);
  int GetPatchesSampled(int max_num_patches, const Classifier& c,
                        std::vector<float>* weights, std::vector<Patch>* patches);

  /**
   * Grab num_patches patches and use them to estimate the average weight,
   * or gradient using classifier c.
   */
  float ComputeAverageWeight(float positive_prob, int num_patches, const Classifier& c);

  /**
   * Read a single patch from the data stream.
   */
  bool ReadPositivePatch(Patch* p);
  bool ReadNegativePatch(Patch* p);

  /**
   * Write a set of patches or patches + labels to a file,
   * for reading later using a DataSource object.
   */
  static void WritePatchesToFile(const std::string& filename, const std::vector<Patch>& patches);
  static void WriteLabeledPatchesToFile(const std::string& filename, const std::vector<Patch>& patches,
        				const std::vector< std::vector<Label> >& labels);

  /**
   * Read a set of patches or patches + labels from a file.
   */
  static int ReadPatchesFromFile(const std::string& filename, int num_patches, std::vector<Patch>* patches);
  static int ReadLabeledPatchesFromFile(const std::string& filename, int num_patches, std::vector<Patch>* patches,
                                        std::vector< std::vector<Label> >* labels);

  void set_num_positives_to_sample(int num) {num_positives_to_sample_ = num;}
  void set_num_negatives_to_sample(int num) {num_negatives_to_sample_ = num;}
  int num_positives_to_sample() {return num_positives_to_sample_;}
  int num_negatives_to_sample() {return num_negatives_to_sample_;}

protected:
  int GetPatchesSampled(float positive_prob, int max_num_patches, float normalizer, const Classifier& c,
        		std::vector<float>* weights, std::vector<Patch>* patches);

  void OpenNextFile(std::vector<std::string>* filenames, int* index, std::ifstream* file);
  bool ReadPositivePatchAttempt(Patch *p);
  bool ReadNegativePatchAttempt(Patch *p);

  bool CheckDataAgainstFlags(bool positive);

  bool frames_mode_;
  
  std::vector<std::string> positive_filenames_;
  std::vector<std::string> negative_filenames_;

  std::ifstream positive_file_;
  std::ifstream negative_file_;

  int positive_filenames_index_;
  int negative_filenames_index_;
  
  int num_positives_to_sample_;
  int num_negatives_to_sample_;
};

  // int GetPositivePatchesSampled(int max_num_patches, const Classifier& c,
  //       			std::vector<float>* weights, std::vector<Patch>* patches);
  // int GetNegativePatchesSampled(int max_num_patches, const Classifier& c,
  //       			std::vector<float>* weights, std::vector<Patch>* patches);

  // int GetPatchesSampled(int max_num_patches, bool positive, const Classifier& c,
  //       		std::vector<float>* weights, std::vector<Patch>* patches);
  // int GetAllPatchesSampled(int max_num_patches, const Classifier& c,
  //       		   std::vector<float>* weights, std::vector<Patch>* patches);

}  // namespace speedboost

#endif  // ifndef SPEEDBOOST_DATA_SOURCE_H
