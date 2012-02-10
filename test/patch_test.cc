//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <fstream>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <iostream>

#include "common.h"
#include "patch.h"
#include "patch.pb.h"

using namespace std;
using namespace speedboost;

class PatchTest : public testing::Test {
protected:
  virtual void SetUp() {
    original = Patch(0, 10, 10, 2);
    float v = 0.0;
    for (int w = 0; w < original.width(); w++) {
      for (int h = 0; h < original.height(); h++) {
        original.SetValue(w, h, 0, v);
        original.SetValue(w, h, 1, 2.0*v);
        v += 1.0;
      }
    }
  }

  Patch original;
};

TEST_F(PatchTest, SimpleTest) {
  Patch empty;

  EXPECT_EQ(empty.width(), 0);
  EXPECT_EQ(empty.height(), 0);
  EXPECT_EQ(empty.channels(), 0);
  
  Patch test(1, 10, 20, 2);

  EXPECT_EQ(test.width(), 10);
  EXPECT_EQ(test.height(), 20);
  EXPECT_EQ(test.channels(), 2);

  test.SetValue(3, 4, 1, 5.0);
  EXPECT_EQ(test.Value(3, 4, 1), 5.0);

  float v = 0.0;
  for (int w = 0; w < original.width(); w++) {
    for (int h = 0; h < original.height(); h++) {
      EXPECT_FLOAT_EQ(v, original.Value(w, h, 0));
      EXPECT_FLOAT_EQ(2*v, original.Value(w, h, 1));
      v += 1.0;
    }
  }
}

TEST_F(PatchTest, IntegralTest) {
  Patch integral(original);
  integral.ComputeIntegralImage();
  for (int w = 0; w < original.width(); w++) {
    for (int h = 0; h < original.height(); h++) {
      // Manually verify sum.
      float sum0 = 0.0;
      float sum1 = 0.0;
      for (int i = 0; i <= w; i++) {
        for (int j = 0; j <= h; j++) {
          sum0 += original.Value(i, j, 0);
          sum1 += original.Value(i, j, 1);
        }
      }

      EXPECT_FLOAT_EQ(sum0, integral.Value(w, h, 0));
      EXPECT_FLOAT_EQ(sum1, integral.Value(w, h, 1));
      EXPECT_FLOAT_EQ(2.0 * integral.Value(w, h, 0), integral.Value(w, h, 1));
    }
  }
}

TEST_F(PatchTest, ResizeTest) {
}

TEST_F(PatchTest, ReadWriteTest) {
  const string kFilename = FLAGS_test_output_directory + "/patch_test_scratch";

  ofstream out(kFilename.c_str(), ofstream::out | ofstream::trunc);
  original.Write(out);
  out.close();

  Patch copy;
  ifstream in(kFilename.c_str(), ifstream::in);
  copy.Read(in);
  in.close();

  EXPECT_EQ(original.width(), copy.width());
  EXPECT_EQ(original.height(), copy.height());
  EXPECT_EQ(original.channels(), copy.channels());
  float v = 0.0;
  for (int w = 0; w < copy.width(); w++) {
    for (int h = 0; h < copy.height(); h++) {
      EXPECT_FLOAT_EQ(v, copy.Value(w, h, 0));
      EXPECT_FLOAT_EQ(2*v, copy.Value(w, h, 1));
      v += 1.0;
    }
  }
}
