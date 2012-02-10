//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "classifier.h"
#include "data_source.h"
#include "util.h"

using namespace std;

DEFINE_int32(num_negatives_to_sample, 50000,
	     "Number of negative patches to sample from (approximately), "
             "e.g. the number of negatives in the data set.");
DEFINE_int32(num_positives_to_sample, 10000,
	     "Number of positive patches to sample from (approximately), "
             "e.g. the number of positives in the data set.");
DEFINE_int32(max_read_attempts, 10,
	     "Max number of attempts at reading (and extracting) a patch before failing.");

namespace speedboost {

DataSource::DataSource(const string& positive_file_glob, const string& negative_file_glob)
  : frames_mode_(false),
    positive_filenames_(),
    negative_filenames_(),
    positive_file_(),
    negative_file_(),
    positive_filenames_index_(0),
    negative_filenames_index_(0),
    num_positives_to_sample_(FLAGS_num_positives_to_sample),
    num_negatives_to_sample_(FLAGS_num_negatives_to_sample) {
  ExpandFileGlob(positive_file_glob, &positive_filenames_);
  ExpandFileGlob(negative_file_glob, &negative_filenames_);

  // Permute the filenames for the first time.
  random_shuffle(positive_filenames_.begin(), positive_filenames_.end());
  random_shuffle(negative_filenames_.begin(), negative_filenames_.end());

  assert(positive_filenames_.size() > 0);
  assert(negative_filenames_.size() > 0);

  assert(CheckDataAgainstFlags(true));
  assert(CheckDataAgainstFlags(false));
}

DataSource::DataSource(const string& frames_file_glob)
  : frames_mode_(true),
    positive_filenames_(),
    negative_filenames_(),
    positive_file_(),
    negative_file_(),
    positive_filenames_index_(0),
    negative_filenames_index_(0),
    num_positives_to_sample_(FLAGS_num_positives_to_sample),
    num_negatives_to_sample_(FLAGS_num_negatives_to_sample) {
  ExpandFileGlob(frames_file_glob, &positive_filenames_);

  // Permute the filenames for the first time.
  random_shuffle(positive_filenames_.begin(), positive_filenames_.end());

  assert(positive_filenames_.size() > 0);
}

int DataSource::GetPositivePatches(int max_num_patches, vector<Patch>* patches) {
  int num_read = 0;
  while (num_read < max_num_patches) {
    Patch p;
    if (!ReadPositivePatch(&p)) {
      return num_read;
    }
    
    patches->push_back(p);
    num_read++;
  }

  return num_read;
}

int DataSource::GetNegativePatches(int max_num_patches, vector<Patch>* patches) {
  int num_read = 0;
  while (num_read < max_num_patches) {
    Patch p;
    if (!ReadNegativePatch(&p)) {
      return num_read;
    }
    
    patches->push_back(p);
    num_read++;
  }

  return num_read;
}

int DataSource::GetPositivePatchesActive(int max_num_patches, const Classifier& c, vector<Patch>* patches) {
  int num_read = 0;
  int num_added = 0;
  while (num_added < max_num_patches) {
    Patch p;
    if (!ReadPositivePatch(&p)) {
      return num_added;
    }

    if (c.IsActiveInLastChain(p)) {
      patches->push_back(p);
      num_added++;
    }
    num_read++;
  }

  cout << "Loaded " << num_added << " patches, read " << num_read << endl;
  return num_added;
}

int DataSource::GetNegativePatchesActive(int max_num_patches, const Classifier& c, vector<Patch>* patches) {
  int num_read = 0;
  int num_added = 0;
  while (num_added < max_num_patches) {
    Patch p;
    if (!ReadNegativePatch(&p)) {
      return num_added;
    }

    if (c.IsActiveInLastChain(p)) {
      patches->push_back(p);
      num_added++;
    }
    num_read++;
  }

  cout << "Loaded " << num_added << " patches, read " << num_read << endl;
  return num_added;
}

int DataSource::GetPositivePatchesSampled(int max_num_patches, const Classifier& c,
                                          vector<float>* weights, vector<Patch>* patches) {
  // Compute normalizer assuming average data set weight.
  float average_weight = ComputeAverageWeight(1.0, 500, c);
  float normalizer = average_weight * (float)num_positives_to_sample_ / (float)max_num_patches;

  cout << "Getting positive patches. avg weight: " << average_weight << " normalizer: " << normalizer << endl;
  return GetPatchesSampled(1.0, max_num_patches, normalizer, c, weights, patches);
}

int DataSource::GetNegativePatchesSampled(int max_num_patches, const Classifier& c,
                                          vector<float>* weights, vector<Patch>* patches) {
  // Compute normalizer assuming average data set weight.
  float average_weight = ComputeAverageWeight(0.0, 500, c);
  float normalizer = average_weight * (float)num_negatives_to_sample_ / (float)max_num_patches;

  cout << "Getting negative patches. avg weight: " << average_weight << " normalizer: " << normalizer << endl;
  return GetPatchesSampled(0.0, max_num_patches, normalizer, c, weights, patches);
}

int DataSource::GetPatchesSampled(int max_num_patches, const Classifier& c,
                                  vector<float>* weights, vector<Patch>* patches) {
  float prob = (float)num_positives_to_sample_ / (float)(num_negatives_to_sample_ + num_positives_to_sample_);

  // Compute normalizer assuming average data set weight.
  float average_weight = ComputeAverageWeight(prob, 500, c);
  float normalizer = average_weight * (float)(num_negatives_to_sample_ + num_positives_to_sample_) / (float)max_num_patches;

  cout << "Getting all patches. avg weight: " << average_weight << " normalizer: " << normalizer << endl;
  return GetPatchesSampled(prob, max_num_patches, normalizer, c, weights, patches);
}

float DataSource::ComputeAverageWeight(float positive_prob, int num_patches, const Classifier& c) {
  int num_read = 0;
  float sum = 0.0;

  while (num_read < num_patches) {
    Patch p;
    
    // Flip coin for positive or negative.
    float sample = (float)rand() / (float)RAND_MAX;

    if (sample < positive_prob) {
      if (!ReadPositivePatch(&p)) {
        break;
      }
    } else {
      if (!ReadNegativePatch(&p)) {
        break;
      }
    }
    num_read++;

    float y = (p.label() > 0) ? 1.0 : -1.0;
    float w = exp(-y * c.Activation(p));
    sum += w;
  }

  return sum / (float)num_read;
}

int DataSource::GetPatchesSampled(float positive_prob, int max_num_patches, float normalizer, const Classifier& c,
                                  vector<float>* weights, vector<Patch>* patches) {
  int num_read_positive = 0;
  int num_read_negative = 0;
  int num_read = 0;
  int num_added = 0;
  float remainder = normalizer * (float)rand() / (float)RAND_MAX;

  while (num_added < max_num_patches) {
    Patch p;
    
    // Flip coin for positive or negative.
    float sample = (float)rand() / (float)RAND_MAX;

    if (sample < positive_prob) {
      if (!ReadPositivePatch(&p)) {
        return num_added;
      }
      num_read_positive++;
    } else {
      if (!ReadNegativePatch(&p)) {
        return num_added;
      }
      num_read_negative++;
    }
    num_read++;

    float y = (p.label() > 0) ? 1.0 : -1.0;
    float w = exp(-y * c.Activation(p));
    if (w + remainder > normalizer) {
      // Number of times the low variance resampler 'hit' this sample.
      float hits = floor((w + remainder) / normalizer);
      
      patches->push_back(p);
      weights->push_back(hits / w);
      remainder = fmod(w + remainder, normalizer);

      num_added++;
    } else {
      remainder += w;
    }
  }

  cout << "Loaded " << num_added << " patches, read " << num_read << endl;
  cout << "positives read: " << num_read_positive << ", negatives: " << num_read_negative
       << ", positive probability: " << positive_prob << endl;
  return num_added;
}

void DataSource::OpenNextFile(vector<string>* filenames, int* index, ifstream* file) {
  if (*index >= (int)(filenames->size())) {
    *index = 0;
    random_shuffle(filenames->begin(), filenames->end());
  }

  if (file->is_open())
    file->close();

  file->open((*filenames)[*index].c_str());
  (*index)++;
}

bool DataSource::ReadPositivePatchAttempt(Patch *p) {
  if (frames_mode_) {
    cout << "Frames mode not supported yet." << endl;
    return false;
  } else {
    if (!positive_file_.good()) {
      OpenNextFile(&positive_filenames_, &positive_filenames_index_, &positive_file_);
    }

    return p->Read(positive_file_);
  }
}

bool DataSource::ReadNegativePatchAttempt(Patch *p) {
  if (frames_mode_) {
    cout << "Frames mode not supported yet." << endl;
    return false;
  } else {
    if (!negative_file_.good()) {
      OpenNextFile(&negative_filenames_, &negative_filenames_index_, &negative_file_);
    }
    
    return p->Read(negative_file_);
  }
}

bool DataSource::ReadPositivePatch(Patch* p) {
  for (int i = 0; i < FLAGS_max_read_attempts; i++) {
    if (ReadPositivePatchAttempt(p)) {
      p->ComputeIntegralImage();
      return true;
    }
  }
  return false;
}

bool DataSource::ReadNegativePatch(Patch* p) {
  for (int i = 0; i < FLAGS_max_read_attempts; i++) {
    if (ReadNegativePatchAttempt(p)) {
      p->ComputeIntegralImage();
      return true;
    }
  }
  return false;
}

bool DataSource::CheckDataAgainstFlags(bool positive) {
  Patch p;
  if (positive) {
    ReadPositivePatch(&p);
  } else {
    ReadNegativePatch(&p);
  }

  // Check the size against current flag
  if (p.width() != FLAGS_patch_width) {
    if (google::GetCommandLineFlagInfoOrDie("patch_width").is_default) {
      cout << "WARNING: changing patch_width flag from default of " << FLAGS_patch_width
           << " to " << p.width() << " to match input data." << endl;
      FLAGS_patch_width = p.width();
    } else {
      cout << "ERROR: patch_width specified in flags differs from patch data sizes" << endl;
      return false;
    }
  }
  
  if (p.height() != FLAGS_patch_height) {
    if (google::GetCommandLineFlagInfoOrDie("patch_height").is_default) {
      cout << "WARNING: changing patch_height flag from default of " << FLAGS_patch_height
           << " to " << p.height() << " to match input data." << endl;
      FLAGS_patch_height = p.height();
    } else {
      cout << "ERROR: patch_height specified in flags differs from patch data sizes" << endl;
      return false;
    }
  }
  
  if (p.channels() != FLAGS_patch_depth) {
    if (google::GetCommandLineFlagInfoOrDie("patch_depth").is_default) {
      cout << "WARNING: changing patch_depth flag from default of " << FLAGS_patch_depth
           << " to " << p.channels() << " to match input data." << endl;
      FLAGS_patch_depth = p.channels();
    } else {
      cout << "ERROR: patch_depth specified in flags differs from patch data sizes" << endl;
      return false;
    }
  }

  return true;
}

void DataSource::WritePatchesToFile(const string& filename, const vector<Patch>& patches) {
  ofstream out(filename.c_str(), ofstream::out);
  for (unsigned int i = 0; i < patches.size(); i++) {
    patches[i].Write(out);
  }
  out.close();
}

void DataSource::WriteLabeledPatchesToFile(const string& filename, const vector<Patch>& patches,
					   const vector< vector<Label> >& labels) {
  ofstream out(filename.c_str(), ofstream::out);
  for (unsigned int i = 0; i < patches.size(); i++){
    patches[i].Write(out);
    
    int num_labels = labels[i].size();
    out.write((char*)(&num_labels), sizeof(int));
    for (int j = 0; j < num_labels; j++) {
      labels[i][j].Write(out);
    }
  }
  out.close();
}

int DataSource::ReadPatchesFromFile(const string& filename, int max_num_patches, vector<Patch>* patches) {
  ifstream in(filename.c_str(), ifstream::in);

  int num_read = 0;
  while (in.good() && (num_read < max_num_patches)) {
    Patch p;
    if (p.Read(in)) {
      patches->push_back(p);
      num_read++;
    }
  }
  in.close();

  return num_read;
}

int DataSource::ReadLabeledPatchesFromFile(const string& filename, int max_num_patches, vector<Patch>* patches,
                                           vector< vector<Label> >* labels) {
  ifstream in(filename.c_str(), ifstream::in);

  int num_read = 0;
  while (in.good() && (num_read < max_num_patches)) {
    Patch p;
    if (p.Read(in)) {
      patches->push_back(p);
      labels->push_back(vector<Label>());

      int num_labels = 0;
      in.read((char*)(&num_labels), sizeof(int));
      for (int j = 0; j < num_labels; j++) {
	Label l;
	l.Read(in);
        labels->back().push_back(l);
      }

      num_read++;
    }
  }
  in.close();

  return num_read;
}

}  // namespace speedboost
