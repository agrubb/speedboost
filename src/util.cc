//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#include <glob.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <fstream>

#include "util.h"

using namespace std;
using namespace google::protobuf;
using namespace google::protobuf::io;

namespace speedboost {

void ExpandFileGlob(const string& pattern, vector<string>* filenames) {
  glob_t glob_data;

  if (glob(pattern.c_str(), 0, NULL, &glob_data)) {
    return;
  }

  for (int i = 0; i < (int)(glob_data.gl_pathc); i++) {
    filenames->push_back(string(glob_data.gl_pathv[i]));
  }

  globfree(&glob_data);  
}

bool ReadMessage(CodedInputStream* in, Message* msg) {
  string input_string;
  unsigned int input_len;

  if (!in->ReadVarint32(&input_len))
    return false;

  input_string.resize(input_len);
  if (!in->ReadString(&input_string, input_len))
    return false;

  return msg->ParseFromString(input_string);
}

bool ReadMessage(istream& in, Message* msg) {
  string input_string;
  unsigned int input_len;
  
  if (!in.good()) return false;

  in.read((char*)(&input_len), sizeof(int));
  if (!in.good()) return false;

  input_string.resize(input_len);
  in.read(const_cast<char *>(input_string.c_str()), input_len);
  if (!in.good()) return false;

  return msg->ParseFromString(input_string);
}

bool ReadMessageFromFileAsText(const string& filename, Message* msg) {
  ifstream in(filename.c_str(), ifstream::in);
  IstreamInputStream* input = new IstreamInputStream(&in);

  bool ret = TextFormat::Parse(input, msg);
  
  delete input;
  return ret;
}

void WriteMessage(CodedOutputStream* out, const Message& msg) {
  string output_string;
  unsigned int output_len;

  msg.SerializeToString(&output_string);

  output_len = output_string.length();
  out->WriteVarint32(output_len);
  out->WriteString(output_string);
}

void WriteMessage(ostream& out, const Message& msg) {
  string output_string;
  unsigned int output_len;

  msg.SerializeToString(&output_string);

  output_len = output_string.length();
  out.write((char*)(&output_len), sizeof(unsigned int));
  out.write(const_cast<char *>(output_string.c_str()), output_len);
}

bool WriteMessageToFileAsText(const string& filename, const Message& msg) {
  ofstream out(filename.c_str(), ofstream::out | ofstream::trunc);
  OstreamOutputStream* output = new OstreamOutputStream(&out);

  if (!out.good())
    return false;

  TextFormat::Print(msg, output);
  
  delete output;
  return true;
}

}  // namespace speedboost
