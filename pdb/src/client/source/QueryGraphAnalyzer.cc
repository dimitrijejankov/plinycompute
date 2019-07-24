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

std::string QueryGraphAnalyzer::parseTCAPString(Vector<Handle<Computation>> &computations) {

  // clear all the markers
  clearGraph();

  // we start with the label 0 for the computation
  int computationLabel = 0;

  // we pull al the partial TCAP strings here
  std::vector<std::string> TCAPStrings;

  // go through each sink
  for (int i = 0; i < this->queryGraph.size(); i++) {

    // traverse the graph, this basically adds all the visited child computations of the graph in the order they are labeled
    // and gives us the partial TCAP strings
    std::vector<InputTupleSetSpecifier> inputTupleSets;
    queryGraph[i]->traverse(TCAPStrings, computations, inputTupleSets, computationLabel);

    // add the root computation
    computations.push_back(queryGraph[i]);
  }

  // merge all the strings
  std::string TCAPStringToReturn;
  for (const auto &tcapString : TCAPStrings) {
    TCAPStringToReturn += tcapString;
  }

  // return the TCAP string
  return TCAPStringToReturn;
}

void QueryGraphAnalyzer::clearGraph() {

  // go through each sink and clear
  for (const auto &sink : this->queryGraph) {
    sink->clearGraph();
  }
}

}

#endif
