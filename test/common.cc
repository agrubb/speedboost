//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <gflags/gflags.h>

DEFINE_string(test_output_directory, "tmp",
              "Directory for creating temporary files in during tests.");

DEFINE_string(test_data_directory, "test/data",
              "Directory where test data is stored (for testing detector, etc.).");
