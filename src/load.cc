//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <iostream>
#include <fstream>
#include <gflags/gflags.h>
#include <ImageMagick/Magick++.h>
#include <sstream>
#include <string>
#include <vector>

#include "data_source.h"
#include "image_util.h"
#include "patch.h"

using namespace speedboost;
using namespace std;

DEFINE_string(label_filename, "",
              "Label file containing images and labeled patches.");
DEFINE_string(output_filename, "",
              "File to save patch data to.");
DEFINE_int32(label, 0,
	     "Label to give patches, e.g. 0 for negative, 1 for positive.");
DEFINE_bool(extract_patches, true,
	    "If true, extract the labeled patches.  If false, simply store"
	    "images and labels for later extraction.");
DEFINE_bool(output_images, false,
	    "If output_frames is true, write all the loaded images to output_images_directory.");
DEFINE_string(output_images_directory, "",
	      "If output_frames is true, write images as [output_frames_directory]/[index].ppm.");

bool ParseLabelsAndPatches(string filename, vector<Patch>* patches, vector< vector<Label> >* labels) {
  ifstream file(filename.c_str());
  if ( !file.is_open() )
    return false;

  // load files relative to the label file
  size_t pos = filename.rfind('/');
  string dirname = pos == string::npos ? "" : filename.substr(0, pos) + "/";
  while( !file.eof() ) {
    string str;
    file >> str;
    if (str.empty()) break;
    if (str.at(0) == '#' ) continue; /* comment */

    // cout << dirname + str << endl;

    Magick::Image frame(dirname + str);
    if (FLAGS_patch_depth == 3) {
      frame.type(Magick::TrueColorType);
    } else {
      frame.type(Magick::GrayscaleType);
    }

    Patch p(FLAGS_label, frame.columns(), frame.rows(), FLAGS_patch_depth);
    ImageToPatch(frame, &p);
    patches->push_back(p);

    int num_labels = 0;
    file >> num_labels;
    vector<Label> patch_labels;
    for (int i = 0; i < num_labels; i++) {
      int x, y, w, h;
      file >> x >> y >> w >> h;
      // cout << "Label @ (" << x << ", " << y << ") " << w << "x" << h << endl;
      
      patch_labels.push_back(Label(x, y, w, h, 1));
    }
    labels->push_back(patch_labels);
  }
  file.close();

  return true;
}

bool OutputImages(string filename, const vector<Patch>& patches) {
  for (unsigned int i = 0; i < patches.size(); i++) {
    size_t pos = filename.rfind('/');
    string dirname = pos == string::npos ? filename : filename.substr(0, pos);
    stringstream ss;
    ss << dirname << "/" << i << ".ppm";
    string filename = ss.str();
    patches[i].WritePPM(filename);
  }

  return true;
}

int main(int argc, char *argv[])
{
  string usage = "Load a set of labeled images into binary patch format.  Usage:\n";
  usage += argv[0];
  usage += " [options]";
  google::SetUsageMessage(usage);

  // parse up the flags
  google::ParseCommandLineFlags(&argc, &argv, true);

  if ("" == FLAGS_label_filename) {
    cout << "Label filename is empty, exiting." << endl;
    google::ShowUsageWithFlags(argv[0]);
    return 1;
  }

  if ("" == FLAGS_output_filename) {
     cout << "Output filename is empty, exiting." << endl;
     google::ShowUsageWithFlags(argv[0]);
     return 1;
  }

  vector<Patch> frames;
  vector< vector<Label> > labels;
  ParseLabelsAndPatches(FLAGS_label_filename, &frames, &labels);

  if (FLAGS_extract_patches) {
    vector<Patch> patches;
    for (unsigned int i = 0; i < frames.size(); i++) {
      for (unsigned int j = 0; j < labels[i].size(); j++) {
	Patch p(FLAGS_label, FLAGS_patch_width, FLAGS_patch_height, FLAGS_patch_depth);
	frames[i].ExtractLabel(labels[i][j], &p);
	patches.push_back(p);
      }
    }

    DataSource::WritePatchesToFile(FLAGS_output_filename, patches);
    if (FLAGS_output_images) {
      OutputImages(FLAGS_output_images_directory, patches);
    }
  } else {
    DataSource::WriteLabeledPatchesToFile(FLAGS_output_filename, frames, labels);
    if (FLAGS_output_images) {
      OutputImages(FLAGS_output_images_directory, frames);
    }
  }

  return 0;
}
