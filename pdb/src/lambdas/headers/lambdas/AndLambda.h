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
    return std::string("and");
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
  std::string toTCAPString(std::vector<std::string> &childrenLambdaNames,
                           int lambdaLabel,
                           const std::string &computationName,
                           int computationLabel,
                           std::string &myLambdaName,
                           MultiInputsBase *multiInputsComp,
                           bool shouldFilter,
                           const std::string &parentLambdaName,
                           bool isSelfJoin) override {

    // create the data for the lambda
    mustache::data lambdaData;
    lambdaData.set("computationName", computationName);
    lambdaData.set("computationLabel", std::to_string(computationLabel));
    lambdaData.set("typeOfLambda", getTypeOfLambda());
    lambdaData.set("lambdaLabel", std::to_string(lambdaLabel));

    // create the lambda name
    mustache::mustache lambdaNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}"};
    myLambdaName = lambdaNameTemplate.render(lambdaData);

    // create the computation name with label
    mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
    std::string computationNameWithLabel = computationNameWithLabelTemplate.render(lambdaData);

    // grab the pointer to the lhs and rhs
    auto lhsPtr = lhs.getPtr().get();
    auto rhsPtr = rhs.getPtr().get();

    // is the lhs and rhs joined
    bool joined = isJoined(lhsPtr->joinedInputs, rhsPtr->joinedInputs);

    /**
     * 1. If this is an expression and not a predicated, throw an exception since we don't support this currently
     */

    if(!lhsPtr->isFiltered || !rhsPtr->isFiltered) {
      throw runtime_error("We only currently support && predicates, expressions are not supported. ");
    }

    /**
     * 2. Check if this is already joined and filtered if it is, we don't need to do anything
     */

    // check if all the columns are in the same tuple set, in that case we apply the equals lambda directly onto that tuple set
    if (joined && lhsPtr->isFiltered && rhsPtr->isFiltered) {

      // this has to be true
      assert(!lhsPtr->joinedInputs.empty());
      assert(!rhsPtr->joinedInputs.empty());

      // grab the input index, it should not matter whether it is the lhs or rhs
      auto inputIndex = *lhsPtr->joinedInputs.begin();

      outputTupleSetName = multiInputsComp->tupleSetNamesForInputs[inputIndex];

      // copy the inputs to the output
      outputColumns = multiInputsComp->inputColumnsForInputs[inputIndex];

      // we are not applying anything
      appliedColumns.clear();

      // we are not generating
      generatedColumns.clear();

      // mark as filtered
      isFiltered = true;

      // update the join group
      joinGroup = multiInputsComp->joinGroupForInput[inputIndex];

      // go through each tuple set and update stuff
      for(int i = 0; i < multiInputsComp->tupleSetNamesForInputs.size(); ++i) {

        // check if this tuple set is the same index
        if(multiInputsComp->joinGroupForInput[i] == joinGroup) {

          // the output tuple set is the new set with these columns
          multiInputsComp->tupleSetNamesForInputs[i] = outputTupleSetName;
          multiInputsComp->inputColumnsForInputs[i] = outputColumns;
          multiInputsComp->inputColumnsToApplyForInputs[i] = generatedColumns;

          // this input was joined
          joinedInputs.insert(i);
        }
      }

      // not TCAP is generated
      return "";
    }

    /**
     * 3. This is a predicate but it is not joined therefore we need to do a cartasian join there is no need to do a
     * filter after this since the inputs are already filtered and this is an and predicate.
     */


    /**
     * 3.1. Create a hash one for the LHS side
     */

    // get the index of the left input, any will do since all joined tuple sets are the same
    auto lhsIndex = *lhsPtr->joinedInputs.begin();
    auto lhsColumnNames = multiInputsComp->inputColumnsForInputs[lhsIndex];

    // added the lhs attribute
    lambdaData.set("LHSApplyAttribute", lhsColumnNames[0]);

    // we need a cartesian join hash-one for lhs
    const std::string &leftTupleSetName = multiInputsComp->tupleSetNamesForInputs[lhsIndex];

    // the lhs column can be any column we only need it to get the number of rows
    std::vector<std::string> leftColumnsToApply = { lhsColumnNames[0] };

    // make the output tuple set
    mustache::mustache leftOutputTupleTemplate{"hashOneFor_{{LHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}"};
    std::string leftOutputTupleSetName = leftOutputTupleTemplate.render(lambdaData);

    // make the column name
    mustache::mustache leftOutputColumnNameTemplate{"OneFor_left_{{computationLabel}}_{{lambdaLabel}}"};
    std::string leftOutputColumnName = leftOutputColumnNameTemplate.render(lambdaData);

    // make the columns
    std::vector<std::string> leftOutputColumns = lhsColumnNames;
    leftOutputColumns.push_back(leftOutputColumnName);

    // make the lambda
    std::string tcapString = formatLambdaComputation(leftTupleSetName,
                                                     lhsColumnNames,
                                                     leftColumnsToApply,
                                                     leftOutputTupleSetName,
                                                     leftOutputColumns,
                                                     "HASHONE",
                                                     computationNameWithLabel,
                                                     "",
                                                     {});

    /**
     * 3.2 Create a hash one for the RHS side
     */

    // get the index of the left input, any will do since all joined tuple sets are the same
    auto rhsIndex = *rhsPtr->joinedInputs.begin();
    auto rhsColumnNames = multiInputsComp->inputColumnsForInputs[rhsIndex];

    lambdaData.set("RHSApplyAttribute", rhsColumnNames[0]);

    // we need a cartesian join hash-one for rhs
    std::string rightTupleSetName = multiInputsComp->tupleSetNamesForInputs[rhsIndex];

    // the rhs column can be any column we only need it to get the number of rows
    std::vector<std::string> rightColumnsToApply = { rhsColumnNames[0] };

    // make the output tuple set
    mustache::mustache rightOutputTupleSetNameTemplate{"hashOneFor_{{RHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}"};
    std::string rightOutputTupleSetName = rightOutputTupleSetNameTemplate.render(lambdaData);

    // make the column name
    mustache::mustache rightOutputColumnNameTemplate{"OneFor_right_{{computationLabel}}_{{lambdaLabel}}"};
    std::string rightOutputColumnName = rightOutputColumnNameTemplate.render(lambdaData);

    // make the columns
    std::vector<std::string> rightOutputColumns = rhsColumnNames;
    rightOutputColumns.push_back(rightOutputColumnName);

    // make the lambda
    tcapString += formatLambdaComputation(rightTupleSetName,
                                          rhsColumnNames,
                                          rightColumnsToApply,
                                          rightOutputTupleSetName,
                                          rightOutputColumns,
                                          "HASHONE",
                                          computationNameWithLabel,
                                          "",
                                          {});

    /**
     * 3.3 Make the cartasian join
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
                                        computationNameWithLabel);

    // update the fields
    isFiltered = true;
    appliedColumns = {};
    generatedColumns = {};

    // go through each tuple set and update stuff
    for(int i = 0; i < multiInputsComp->tupleSetNamesForInputs.size(); ++i) {

      // check if this tuple set is the same index
      if (multiInputsComp->joinGroupForInput[i] == multiInputsComp->joinGroupForInput[lhsIndex] ||
          multiInputsComp->joinGroupForInput[i] == multiInputsComp->joinGroupForInput[rhsIndex] ) {

        // the output tuple set is the new set with these columns
        multiInputsComp->tupleSetNamesForInputs[i] = outputTupleSetName;
        multiInputsComp->inputColumnsForInputs[i] = outputColumns;
        multiInputsComp->inputColumnsToApplyForInputs[i] = generatedColumns;

        // this input was joined
        joinedInputs.insert(i);

        // update the join group so that rhs has the same group as lhs
        multiInputsComp->joinGroupForInput[i] = multiInputsComp->joinGroupForInput[lhsIndex];
      }
    }

    // update the join group
    joinGroup = multiInputsComp->joinGroupForInput[lhsIndex];

    return tcapString;
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