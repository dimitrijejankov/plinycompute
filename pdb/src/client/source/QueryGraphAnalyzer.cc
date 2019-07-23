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

QueryGraphAnalyzer::QueryGraphAnalyzer(const vector<Handle < Computation>> &queryGraph) {
  for (const auto &i : queryGraph) {
    this->queryGraph.push_back(i);
  }
}

QueryGraphAnalyzer::QueryGraphAnalyzer(Handle<Vector<Handle<Computation>>> queryGraph) {
  for (int i = 0; i < queryGraph->size(); i++) {
    this->queryGraph.push_back((*queryGraph)[i]);
  }
}

std::string QueryGraphAnalyzer::parseTCAPString() {

  Handle<Computation> curSink;
  int computationLabel = 0;
  std::vector<std::string> tcapStrings;

  // go through each sink
  for (int i = 0; i < this->queryGraph.size(); i++) {

    std::vector<InputTupleSetSpecifier> inputTupleSets;
    InputTupleSetSpecifier inputTupleSet;
    inputTupleSets.push_back(inputTupleSet);
    curSink = queryGraph[i];
    std::string outputTupleSetName;
    std::vector<std::string> outputColumnNames;
    std::string addedOutputColumnName;

    // traverse the graph
    traverse(tcapStrings, curSink, inputTupleSets, computationLabel,
             outputTupleSetName, outputColumnNames, addedOutputColumnName);
  }


  std::string tcapStringToReturn;
  for (const auto &tcapString : tcapStrings) {
    tcapStringToReturn += tcapString;
  }

  std::cout << tcapStringToReturn << std::endl;
  return tcapStringToReturn;
}

void QueryGraphAnalyzer::traverse(std::vector<std::string> &tcapStrings,
                                  const Handle<Computation>& comp,
                                  const std::vector<InputTupleSetSpecifier>& inputTupleSets,
                                  int &computationLabel,
                                  std::string &outputTupleSetName,
                                  std::vector<std::string> &outputColumnNames,
                                  std::string &addedOutputColumnName) {

  // if there are not inputs to computation, we just process it here
  if(comp->getNumInputs() == 0) {

    // this is a scan set do stuff...
    if (!comp->isTraversed()) {

      outputColumnNames.clear();
      addedOutputColumnName = "";
      std::string curTCAPString = comp->toTCAPString(inputTupleSets, computationLabel);
      tcapStrings.push_back(curTCAPString);
      computationLabel++;
    }

    // get the output tuple set and the column
    outputTupleSetName = comp->getOutputTupleSetName();
    addedOutputColumnName = comp->getOutputColumnToApply();
    outputColumnNames = { addedOutputColumnName };

    // we are out of here
    return;
  }

  // so if the computation is not a scan set, meaning it has at least one input process the children first
  // go through each child and traverse them
  std::vector<InputTupleSetSpecifier> inputTupleSetsForMe;
  for (int i = 0; i < comp->getNumInputs(); i++) {

    // get the child computation
    Handle<Computation> childComp = comp->getIthInput(i);

    // if we have not visited this computation visit it
    if (!childComp->isTraversed()) {

      // go traverse the child computation
      traverse(tcapStrings,
               childComp,
               inputTupleSets,
               computationLabel,
               outputTupleSetName,
               outputColumnNames,
               addedOutputColumnName);

      // mark the computation as transversed
      childComp->setTraversed(true);
    }

    // we met a computation that we have visited just grab the name of the output tuple set and the columns it has
    outputTupleSetName = childComp->getOutputTupleSetName();
    addedOutputColumnName = childComp->getOutputColumnToApply();
    outputColumnNames = { addedOutputColumnName };

    InputTupleSetSpecifier curOutput(outputTupleSetName, outputColumnNames, { addedOutputColumnName });
    inputTupleSetsForMe.push_back(curOutput);
  }

  outputColumnNames.clear();
  addedOutputColumnName.clear();
  outputTupleSetName.clear();

  // generate the TCAP string for this computation
  std::string curTCAPString = comp->toTCAPString(inputTupleSetsForMe, computationLabel);

  // store the TCAP string generated
  tcapStrings.push_back(curTCAPString);

  // go to the next computation
  computationLabel++;
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
