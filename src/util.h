//
// Copyright 2011 Carnegie Mellon University
//
// @author Alex Grubb (agrubb@cmu.edu)
//

#ifndef SPEEDBOOST_UTIL_H
#define SPEEDBOOST_UTIL_H

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/message.h>

namespace speedboost {

void ExpandFileGlob(const std::string& pattern, std::vector<std::string>* filenames);

/**
 * Read protobuf messages in binary form from iostream or a google input stream.
 */
bool ReadMessage(google::protobuf::io::CodedInputStream* in, google::protobuf::Message* msg);
bool ReadMessage(std::istream& in, google::protobuf::Message* msg);

/**
 * Write protobuf messages in binary form to iostream or a google output stream.
 */
void WriteMessage(google::protobuf::io::CodedOutputStream* out, const google::protobuf::Message& msg);
void WriteMessage(std::ostream& out, const google::protobuf::Message& msg);

/**
 * Read and write messages as human-readable text.
 */
bool ReadMessageFromFileAsText(const std::string& filename, google::protobuf::Message* msg);
bool WriteMessageToFileAsText(const std::string& filename, const google::protobuf::Message& msg);

}  // namespace speedboost

#endif  // ifndef SPEEDBOOST_UTIL_H
