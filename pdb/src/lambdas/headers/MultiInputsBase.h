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
#include <set>
#include <map>
#include <algorithm>

namespace pdb {

// this class represents all inputs for a multi-input computation like Join
// this is used in query graph analysis
class MultiInputsBase {

 public:

  // tuple set names for each input
  std::vector<std::string> tupleSetNamesForInputs;

  // input columns for each input
  std::vector<std::vector<std::string>> inputColumnsForInputs;

  // input column to apply for each input
  std::vector<std::vector<std::string>> inputColumnsToApplyForInputs;

  // lambda names to extract for each input and each predicate
  std::vector<std::map<std::string, std::vector<std::string>>> lambdaNamesForInputs;

  // a list of columns we want from the output of the lambda
  std::set<std::string> inputColumnsToKeep;

  // input names for this join operation
  std::vector<std::string> inputNames;

  // this basically tells us what inputs are joined
  std::vector<int32_t> joinGroupForInput;

  explicit MultiInputsBase(int numInputs) {

    // make the vectors the appropriate size
    tupleSetNamesForInputs.resize(numInputs);
    inputColumnsForInputs.resize(numInputs);
    inputColumnsToApplyForInputs.resize(numInputs);
    lambdaNamesForInputs.resize(numInputs);
    inputNames.resize(numInputs);

    // label each join group
    joinGroupForInput.resize(numInputs);
    for(int i = 0; i < numInputs; ++i) { joinGroupForInput[i] = i; }
  }

  // resizes the multi input
  void resize(int numInputs) {

    // clear all
    inputNames.clear();
    inputColumnsToKeep.clear();
    inputColumnsForInputs.clear();
    inputColumnsToApplyForInputs.clear();

    // make the vectors the appropriate size
    tupleSetNamesForInputs.resize(numInputs);
    inputColumnsForInputs.resize(numInputs);
    inputColumnsToApplyForInputs.resize(numInputs);
    lambdaNamesForInputs.resize(numInputs);
    inputNames.resize(numInputs);
  }

  std::vector<std::string> getNotAppliedInputColumnsForIthInput(int i) {

    // the return value
    std::vector<std::string> ret;
    for(const auto &it : inputColumnsForInputs[i]) {

      // check if it is in input columns to apply
      if(std::find(inputColumnsToApplyForInputs[i].begin(), inputColumnsToApplyForInputs[i].end(), it) == inputColumnsToApplyForInputs[i].end()) {
        ret.emplace_back(it);
      }
    }

    // return the input columns
    return std::move(ret);
  }

};
}

#endif