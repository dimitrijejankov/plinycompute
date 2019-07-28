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

#ifndef AND_LAM_H
#define AND_LAM_H

#include <vector>
#include "Lambda.h"
#include "executors/ComputeExecutor.h"
#include "TupleSetMachine.h"
#include "TypedLambdaObject.h"
#include "TupleSet.h"
#include "Ptr.h"

namespace pdb {

// only one of these four versions is going to work... used to automatically dereference a Ptr<blah>
// type on either the LHS or RHS of an "and" check
template<class LHS, class RHS>
std::enable_if_t<std::is_base_of<PtrBase, LHS>::value && std::is_base_of<PtrBase, RHS>::value, bool> checkAnd(LHS lhs,
                                                                                                              RHS rhs) {
  return *lhs && *rhs;
}

template<class LHS, class RHS>
std::enable_if_t<std::is_base_of<PtrBase, LHS>::value && !(std::is_base_of<PtrBase, RHS>::value),
                 bool> checkAnd(LHS lhs, RHS rhs) {
  return *lhs && rhs;
}

template<class LHS, class RHS>
std::enable_if_t<!(std::is_base_of<PtrBase, LHS>::value) && std::is_base_of<PtrBase, RHS>::value,
                 bool> checkAnd(LHS lhs, RHS rhs) {
  return lhs && *rhs;
}

template<class LHS, class RHS>
std::enable_if_t<!(std::is_base_of<PtrBase, LHS>::value) && !(std::is_base_of<PtrBase, RHS>::value),
                 bool> checkAnd(LHS lhs, RHS rhs) {
  return lhs && rhs;
}

template<class LeftType, class RightType>
class AndLambda : public TypedLambdaObject<bool> {

 public:

  LambdaTree<LeftType> lhs;
  LambdaTree<RightType> rhs;

 public:

  AndLambda(LambdaTree<LeftType> lhsIn, LambdaTree<RightType> rhsIn) {
    lhs = lhsIn;
    rhs = rhsIn;
    this->setInputIndex(0, lhs.getInputIndex(0));
    this->setInputIndex(1, rhs.getInputIndex(0));
  }

  ComputeExecutorPtr getExecutor(TupleSpec& inputSchema,
                                 TupleSpec& attsToOperateOn,
                                 TupleSpec& attsToIncludeInOutput) override {

    // create the output tuple set
    TupleSetPtr output = std::make_shared<TupleSet>();

    // create the machine that is going to setup the output tuple set, using the input tuple set
    TupleSetSetupMachinePtr myMachine = std::make_shared<TupleSetSetupMachine>(inputSchema, attsToIncludeInOutput);

    // these are the input attributes that we will process
    std::vector<int> inputAtts = myMachine->match(attsToOperateOn);
    int firstAtt = inputAtts[0];
    int secondAtt = inputAtts[1];

    // this is the output attribute
    auto outAtt = (int) attsToIncludeInOutput.getAtts().size();

    return std::make_shared<ApplyComputeExecutor>(
        output,
        [=](TupleSetPtr input) {

          // set up the output tuple set
          myMachine->setup(input, output);

          // get the columns to operate on
          std::vector<LeftType>& leftColumn = input->getColumn<LeftType>(firstAtt);
          std::vector<RightType>& rightColumn = input->getColumn<RightType>(secondAtt);

          // create the output attribute, if needed
          if (!output->hasColumn(outAtt)) {
            auto outColumn = new std::vector<bool>;
            output->addColumn(outAtt, outColumn, true);
          }

          // get the output column
          std::vector<bool>& outColumn = output->getColumn<bool>(outAtt);

          // loop down the columns, setting the output
          auto numTuples = leftColumn.size();
          outColumn.resize(numTuples);
          for (int i = 0; i < numTuples; i++) {
            outColumn[i] = checkAnd(leftColumn[i], rightColumn[i]);
          }
          return output;
        });
  }

  std::string getTypeOfLambda() override {
    return std::string("&&");
  }

  int getNumChildren() override {
    return 2;
  }

  unsigned int getNumInputs() override {
    return 2;
  }

  LambdaObjectPtr getChild(int which) override {
    if (which == 0)
      return lhs.getPtr();
    if (which == 1)
      return rhs.getPtr();
    return nullptr;
  }

  //TODO: add comment here and refractor the code
  std::string toTCAPString(std::vector<std::string> &inputTupleSetNames,
                           std::vector<std::string> &inputColumnNames,
                           std::vector<std::string> &inputColumnsToApply,
                           std::vector<std::string> &childrenLambdaNames,
                           int lambdaLabel,
                           std::string computationName,
                           int computationLabel,
                           std::string &outputTupleSetName,
                           std::vector<std::string> &outputColumns,
                           std::string &outputColumnName,
                           std::string &myLambdaName,
                           MultiInputsBase *multiInputsComp = nullptr,
                           bool amIPartOfJoinPredicate = false,
                           bool amILeftChildOfEqualLambda = false,
                           bool amIRightChildOfEqualLambda = false,
                           std::string parentLambdaName = "",
                           bool isSelfJoin = false) override {

    if ((multiInputsComp != nullptr) && amIPartOfJoinPredicate) {

      // create the data for the lambda
      mustache::data lambdaData;
      lambdaData.set("computationName", computationName);
      lambdaData.set("computationLabel", std::to_string(computationLabel));
      lambdaData.set("lambdaLabel", std::to_string(lambdaLabel));

      mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
      std::string myComputationName = computationNameWithLabelTemplate.render(lambdaData);

      // Step 1. get list of input names in LHS
      unsigned int leftIndex = lhs.getInputIndex(0);
      std::vector<std::string> lhsColumnNames = multiInputsComp->getInputColumnsForIthInput(leftIndex);
      std::vector<std::string> lhsInputNames;
      for (const auto &curColumnName : lhsColumnNames) {
        for (int j = 0; j < multiInputsComp->getNumInputs(); j++) {
          if (multiInputsComp->getNameForIthInput(j) == curColumnName) {
            lhsInputNames.push_back(curColumnName);
            break;
          }
        }
      }

      // Step 2. get list of input names in RHS
      unsigned int rightIndex = rhs.getInputIndex(0);
      std::vector<std::string> rhsColumnNames = multiInputsComp->getInputColumnsForIthInput(rightIndex);
      std::vector<std::string> rhsInputNames;
      for (const auto &curColumnName : rhsColumnNames) {
        for (int j = 0; j < multiInputsComp->getNumInputs(); j++) {
          if (multiInputsComp->getNameForIthInput(j) == curColumnName) {
            rhsInputNames.push_back(curColumnName);
            break;
          }
        }
      }

      // Step 3. if two lists are disjoint do a cartesian join, otherwise return ""
      std::vector<std::string> inputNamesIntersection;

      for (const auto &lhsInputName : lhsInputNames) {
        for (const auto &rhsInputName : rhsInputNames) {
          if (lhsInputName == rhsInputName) {
            inputNamesIntersection.push_back(lhsInputName);
          }
        }
      }

      if (!inputNamesIntersection.empty()) {

        return "";

      } else {

        /**
         * 1. Create a hash one for the LHS side
         */

        // added the lhs attribute
        lambdaData.set("LHSApplyAttribute", lhsInputNames[0]);

        // we need a cartesian join hash-one for lhs
        std::string leftTupleSetName = multiInputsComp->getTupleSetNameForIthInput(leftIndex);
        std::vector<std::string> leftColumnsToApply = { lhsInputNames[0] };

        mustache::mustache leftOutputTupleTemplate{"hashOneFor_{{LHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}"};
        std::string leftOutputTupleSetName = leftOutputTupleTemplate.render(lambdaData);

        mustache::mustache leftOutputColumnNameTemplate{"OneFor_left_{{computationLabel}}_{{lambdaLabel}}"};
        std::string leftOutputColumnName = leftOutputColumnNameTemplate.render(lambdaData);

        std::vector<std::string> leftOutputColumns = lhsColumnNames;
        leftOutputColumns.push_back(leftOutputColumnName);

        std::string tcapString = formatLambdaComputation(leftTupleSetName,
                                                         lhsColumnNames,
                                                         leftColumnsToApply,
                                                         leftOutputTupleSetName,
                                                         leftOutputColumns,
                                                         leftOutputColumnName,
                                                         "HASHONE",
                                                         myComputationName,
                                                         "",
                                                         {});

        /**
         * 2. Create a hash one for the RHS side
         */

        lambdaData.set("RHSApplyAttribute", rhsInputNames[0]);

        //
        std::string rightTupleSetName = multiInputsComp->getTupleSetNameForIthInput(rightIndex);
        std::vector<std::string> rightColumnsToApply = { rhsInputNames[0] };

        //
        mustache::mustache rightOutputTupleSetNameTemplate{"hashOneFor_{{RHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}"};
        std::string rightOutputTupleSetName = rightOutputTupleSetNameTemplate.render(lambdaData);

        //
        mustache::mustache rightOutputColumnNameTemplate{"OneFor_right_{{computationLabel}}_{{lambdaLabel}}"};
        std::string rightOutputColumnName = rightOutputColumnNameTemplate.render(lambdaData);

        //
        std::vector<std::string> rightOutputColumns = rhsColumnNames;
        rightOutputColumns.push_back(rightOutputColumnName);

        //
        tcapString += formatLambdaComputation(rightTupleSetName,
                                              rhsColumnNames,
                                              rightColumnsToApply,
                                              rightOutputTupleSetName,
                                              rightOutputColumns,
                                              rightOutputColumnName,
                                              "HASHONE",
                                              myComputationName,
                                              "",
                                              {});

        /**
         * 3.
         */

        mustache::mustache outputTupleSetTemplate{"CartesianJoined__{{computationLabel}}_{{lambdaLabel}}"};
        outputTupleSetName = outputTupleSetTemplate.render(lambdaData);

        // copy the output columns
        outputColumns = lhsColumnNames;
        outputColumns.insert(outputColumns.end(), rhsColumnNames.begin(), rhsColumnNames.end());

        // generate the join computation
        tcapString += formatJoinComputation(outputTupleSetName,
                                            outputColumns,
                                            leftOutputTupleSetName,
                                            { leftOutputColumnName },
                                            lhsColumnNames,
                                            rightOutputTupleSetName,
                                            {rightOutputColumnName},
                                            rhsColumnNames,
                                            myComputationName);

        // update multiInputsComp
        for (unsigned int i = 0; i < multiInputsComp->getNumInputs(); i++) {
          std::string curInput = multiInputsComp->getNameForIthInput(i);
          auto iter = std::find(outputColumns.begin(), outputColumns.end(), curInput);
          if (iter != outputColumns.end()) {
            multiInputsComp->setTupleSetNameForIthInput(i, outputTupleSetName);
            multiInputsComp->setInputColumnsForIthInput(i, outputColumns);
            multiInputsComp->setInputColumnsToApplyForIthInput(i, outputColumns);
          }
        }
        return tcapString;
      }

    } else {
      return "";
    }
  }
  /**
  * Returns the additional information about this lambda currently just the lambda type
  * @return the map
  */
  std::map<std::string, std::string> getInfo() override {
    // fill in the info
    return std::map<std::string, std::string>{
        std::make_pair("lambdaType", getTypeOfLambda())
    };
  };


};

}

#endif