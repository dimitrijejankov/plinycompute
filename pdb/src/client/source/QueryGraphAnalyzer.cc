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

#include "QueryGraphAnalyzer.h"
#include "InputTupleSetSpecifier.h"
#include <string.h>
#include <vector>

namespace pdb {

QueryGraphAnalyzer::QueryGraphAnalyzer(const vector<Handle<Computation>> &sinks) {

  // move the sinks
  for (const auto &i : sinks) {
    this->sinks.push_back(i);
  }
}

QueryGraphAnalyzer::QueryGraphAnalyzer(const std::vector<std::tuple<uint64_t, std::string, Handle<Computation>>> &sources,
                                       const vector<Handle<Computation>> &sinks) {

  // move the sinks
  for (const auto &s : sinks) {
    this->sinks.push_back(s);
  }

  // move the sources
  for (const auto &s : sources) {
    this->sources.push_back(s);
  }
}

std::string QueryGraphAnalyzer::parseTCAPString(Vector<Handle<Computation>> &computations) {

  // clear all the markers
  clearGraph();

  // mark all the sources
  markSources();

  // we start with the label 0 for the computation
  int computationLabel = 0;

  // we pull al the partial TCAP strings here
  std::vector<std::string> TCAPStrings;

  // go through each sink
  for (int i = 0; i < this->sinks.size(); i++) {

    // traverse the graph, this basically adds all the visited child computations of the graph in the order they are labeled
    // and gives us the partial TCAP strings
    std::vector<InputTupleSetSpecifier> inputTupleSets;
    sinks[i]->traverse(TCAPStrings, computations, inputTupleSets, computationLabel);

    // add the root computation
    computations.push_back(sinks[i]);
  }

  // merge all the strings
  std::string TCAPStringToReturn;
  for (const auto &tcapString : TCAPStrings) {
    TCAPStringToReturn += tcapString;
  }

  // return the TCAP string
  return TCAPStringToReturn;
}

void QueryGraphAnalyzer::markSources() {

  // mark some computations as sources if necessary
  for (int i = 0; i < this->sources.size(); i++) {

    // get the source
    auto &source = std::get<2>(sources[i]);

    // mark the computation as a source and set an appropriate page set
    source->markSource();
    source->setPageSet(std::get<0>(sources[i]), std::get<1>(sources[i]));
  }
}

std::string QueryGraphAnalyzer::parseTCAPForKeys(Vector<Handle<Computation>> &computations) {

  // clear all the markers
  clearGraph();

  // mark all the sources
  markSources();

  // we start with the label 0 for the computation
  int computationLabel = 0;

  // we pull al the partial TCAP strings here
  std::vector<std::string> TCAPStrings;

  // go through each sink
  for (int i = 0; i < this->sinks.size(); i++) {

    // traverse the graph, this basically adds all the visited child computations of the graph in the order they are labeled
    // and gives us the partial TCAP strings
    std::vector<InputTupleSetSpecifier> inputTupleSets;
    sinks[i]->traverseForKeys(TCAPStrings, computations, inputTupleSets, computationLabel);

    // add the root computation
    computations.push_back(sinks[i]);
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
  for (const auto &sink : this->sinks) {
    sink->clearGraph();
  }
}


}