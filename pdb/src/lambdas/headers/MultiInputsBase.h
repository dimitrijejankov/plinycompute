#include <utility>

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
#ifndef MULTI_INPUTS_BASE
#define MULTI_INPUTS_BASE

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

namespace pdb {

// this class represents all inputs for a multi-input computation like Join
// this is used in query graph analysis
class MultiInputsBase {

 private:
  // tuple set names for each input
  std::vector<std::string> tupleSetNamesForInputs;

  // input columns for each input
  std::vector<std::vector<std::string>> inputColumnsForInputs;

  // input column to apply for each input
  std::vector<std::vector<std::string>> inputColumnsToApplyForInputs;

  // lambda names to extract for each input and each predicate
  std::vector<std::map<std::string, std::vector<std::string>>> lambdaNamesForInputs;

  // input names for this join operation
  std::vector<std::string> inputNames;

  int numInputs;

public:

  void setNumInputs(int value) {
    this->numInputs = value;
  }

  int getNumInputs() {
    return this->numInputs;
  }

  // returns the latest tuple set name that contains the i-th input
  std::string getTupleSetNameForIthInput(int i) {
    if (i >= this->getNumInputs()) {
      return "";
    }
    return tupleSetNamesForInputs[i];
  }

  // set the latest tuple set name that contains the i-th input
  void setTupleSetNameForIthInput(int i, std::string name) {
    if (tupleSetNamesForInputs.size() != numInputs) {
      tupleSetNamesForInputs.resize(numInputs);
    }
    if (i < numInputs) {
      tupleSetNamesForInputs[i] = std::move(name);
    }
  }

  // get latest input columns for the tupleset for the i-th input
  std::vector<std::string> getInputColumnsForIthInput(int i) {
    if (i >= this->getNumInputs()) {
      std::vector<std::string> ret;
      return ret;
    }
    return inputColumnsForInputs[i];
  }

  // set latest input columns for the tupleset for the i-th input
  void setInputColumnsForIthInput(int i, std::vector<std::string> &columns) {
    if (inputColumnsForInputs.size() != numInputs) {
      inputColumnsForInputs.resize(numInputs);
    }
    if (i < numInputs) {
      inputColumnsForInputs[i] = columns;
    }
  }

  // get latest input column to apply for the tupleset for the i-th input
  std::vector<std::string> getInputColumnsToApplyForIthInput(int i) {
    if (i >= this->getNumInputs()) {
      std::vector<std::string> ret;
      return ret;
    }
    return inputColumnsToApplyForInputs[i];
  }

  // set latest input column to apply for the tupleset for the i-th input
  void setInputColumnsToApplyForIthInput(int i, std::vector<std::string> &columnsToApply) {
    if (inputColumnsToApplyForInputs.size() != numInputs) {
      inputColumnsToApplyForInputs.resize(numInputs);
    }
    if (i < numInputs) {
      inputColumnsToApplyForInputs[i] = columnsToApply;
    }
  }

  // set latest input column to apply for the tupleset for the i-th input
  void addColumnToInputColumnsToApplyForIthInput(int i, std::string columnToApply) {
    if (inputColumnsToApplyForInputs.size() != numInputs) {
      inputColumnsToApplyForInputs.resize(numInputs);
    }
    if (i < numInputs) {
      inputColumnsToApplyForInputs[i].clear();
      inputColumnsToApplyForInputs[i].emplace_back(columnToApply);
    }
  }

  // set lambdas for the i-th input, and j-th predicate
  void setLambdasForIthInputAndPredicate(int i,
                                         std::string predicateLambda,
                                         std::string lambdaName) {
    if (lambdaNamesForInputs.size() != numInputs) {
      lambdaNamesForInputs.resize(numInputs);
    }
    if (i < numInputs) {
      lambdaNamesForInputs[i][predicateLambda].push_back(lambdaName);
    }
  }

  // get the name for the i-th input
  std::string getNameForIthInput(int i) {
    if (i >= this->getNumInputs()) {
      return "";
    }
    return inputNames[i];
  }

  // set the name for the i-th input
  void setNameForIthInput(int i, std::string name) {
    if (inputNames.size() != numInputs) {
      inputNames.resize(numInputs);
    }
    if (i < numInputs) {
      inputNames[i] = std::move(name);
    }
  }
};
}

#endif