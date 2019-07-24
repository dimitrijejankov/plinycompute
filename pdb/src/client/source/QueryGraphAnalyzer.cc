/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/
#ifndef QUERY_GRAPH_ANALYZER_SOURCE
#define QUERY_GRAPH_ANALYZER_SOURCE

#include "QueryGraphAnalyzer.h"
#include "InputTupleSetSpecifier.h"
#include <string.h>
#include <vector>

namespace pdb {

QueryGraphAnalyzer::QueryGraphAnalyzer(const vector<Handle<Computation>> &queryGraph) {

  // move the computations
  for (const auto &i : queryGraph) {
    this->queryGraph.push_back(i);
  }
}

std::string QueryGraphAnalyzer::parseTCAPString() {

  Handle<Computation> curSink;
  int computationLabel = 0;
  std::vector<std::string> tcapStrings;

  // go through each sink
  for (int i = 0; i < this->queryGraph.size(); i++) {

    // traverse the graph
    std::vector<InputTupleSetSpecifier> inputTupleSets;
    queryGraph[i]->traverse(tcapStrings, inputTupleSets, computationLabel);
  }

  std::string tcapStringToReturn;
  for (const auto &tcapString : tcapStrings) {
    tcapStringToReturn += tcapString;
  }

  std::cout << tcapStringToReturn << std::endl;
  return tcapStringToReturn;
}

void QueryGraphAnalyzer::clearGraphMarks(Handle<Computation> sink) {

  sink->setTraversed(false);
  int numInputs = sink->getNumInputs();
  for (int i = 0; i < numInputs; i++) {
    Handle<Computation> curNode = sink->getIthInput(i);
    clearGraphMarks(curNode);
  }
}

void QueryGraphAnalyzer::clearGraphMarks() {

  for (const auto &sink : this->queryGraph) {
    clearGraphMarks(sink);
  }
}

void QueryGraphAnalyzer::parseComputations(Vector<Handle<Computation>> &computations, Handle<Computation> sink) {

  int numInputs = sink->getNumInputs();
  for (int i = 0; i < numInputs; i++) {
    Handle<Computation> curNode = sink->getIthInput(i);
    parseComputations(computations, curNode);
  }
  if (!sink->isTraversed()) {
    computations.push_back(sink);
    sink->setTraversed(true);
  }
}

void QueryGraphAnalyzer::parseComputations(Vector<Handle<Computation>> &computations) {
  this->clearGraphMarks();
  for (const auto &sink : this->queryGraph) {
    parseComputations(computations, sink);
  }
}
}

#endif
