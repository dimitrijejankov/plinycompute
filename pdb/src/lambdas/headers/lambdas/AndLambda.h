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

#pragma once

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

  AndLambda(LambdaTree<LeftType> lhsIn, LambdaTree<RightType> rhsIn) {

    // add the children
    children[0] = lhsIn.getPtr();
    children[1] = rhsIn.getPtr();
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

  std::string getTypeOfLambda() const override {
    return std::string("and");
  }

  unsigned int getNumInputs() override {
    return 2;
  }

  /**
   * Generates the TCAP string for the && (and) lambda.
   *
   * @param computationLabel - the index of the computation the lambda belongs to.
   * @param lambdaLabel - the label of the labda (just an integer identifier)
   * @param computationName - so this is how we named the computation, usually type with the identifier,
   *                          we need that to generate the TCAP
   * @param parentLambdaName - the name of the parent lambda to this one, if there is not any it is an empty string
   * @param childrenLambdaNames - the names of the child lambdas
   * @param multiInputsComp - all the inputs sets that are currently there
   * @param isPredicate - is this a predicate and we need to generate a filter?
   * @return - the TCAP string
   */
  std::string generateTCAPString(MultiInputsBase *multiInputsComp,
                                 bool isPredicate) override {

    // create the data for the lambda
    mustache::data lambdaData;
    lambdaData.set("computationName", myComputationName);
    lambdaData.set("computationLabel", std::to_string(myComputationLabel));
    lambdaData.set("typeOfLambda", getTypeOfLambda());
    lambdaData.set("lambdaLabel", std::to_string(myLambdaLabel));
    lambdaData.set("tupleSetMidTag", myPrefix);

    // create the computation name with label
    mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
    std::string computationNameWithLabel = computationNameWithLabelTemplate.render(lambdaData);

    // grab the pointer to the lhs and rhs
    auto lhsPtr = children[0];
    auto rhsPtr = children[1];

    // is the lhs and rhs joined
    bool joined = lhsPtr->joinGroup == rhsPtr->joinGroup;

    /**
     * 0. If this is an expression
     */

    // if this and lambda is not a predicate but an expression that means that we need to apply and equals lambda on it.
    if(!isPredicate) {

      // must be an expression if lhs and rhs are expressions and they must be in the same tuple set (as all expressions must be)
      assert(!lhsPtr->isFiltered);
      assert(!rhsPtr->isFiltered);
      assert(joined);

      // must have exactly one generated column in both rhs and lhs
      assert(lhsPtr->generatedColumns.size() == 1);
      assert(rhsPtr->generatedColumns.size() == 1);

      // get the input index any will do
      auto inputIndex =*lhsPtr->joinedInputs.begin();

      // generate output tupleset name
      mustache::mustache outputTupleSetNameTemplate{"{{tupleSetMidTag}}_and{{lambdaLabel}}_bool_{{computationName}}{{computationLabel}}"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // create the output column name
      mustache::mustache outputColumnNameTemplate{"and_{{lambdaLabel}}_{{computationLabel}}_bool"};
      std::string outputColumnName = outputColumnNameTemplate.render(lambdaData);

      // remove the lhs input if it is not the original input column
      auto inputs = multiInputsComp->inputColumnsForInputs[inputIndex];
      if(std::find(multiInputsComp->inputNames.begin(), multiInputsComp->inputNames.end(), lhsPtr->generatedColumns[0]) == multiInputsComp->inputNames.end()) {
        inputs.erase(std::remove(inputs.begin(), inputs.end(), lhsPtr->generatedColumns[0]), inputs.end());
      }

      // remove the rhs input if it is not in the original input colum
      if(std::find(multiInputsComp->inputNames.begin(), multiInputsComp->inputNames.end(), rhsPtr->generatedColumns[0]) == multiInputsComp->inputNames.end()) {
        inputs.erase(std::remove(inputs.begin(), inputs.end(), rhsPtr->generatedColumns[0]), inputs.end());
      }

      // the output are the forwarded inputs with the generated column
      outputColumns = inputs;
      outputColumns.push_back(outputColumnName);

      // the the columns we have applied in this lambda
      appliedColumns = { lhsPtr->generatedColumns[0], rhsPtr->generatedColumns[0] };

      // we are going to be applying the generated boolean column
      generatedColumns = { outputColumnName };

      // update the join group
      joinGroup = multiInputsComp->joinGroupForInput[inputIndex];

      // generate the and lambda
      std::string tcapString = formatLambdaComputation(multiInputsComp->tupleSetNamesForInputs[inputIndex],
                                                       inputs,
                                                       appliedColumns,
                                                       outputTupleSetName,
                                                       outputColumns,
                                                       "APPLY",
                                                       computationNameWithLabel,
                                                       getLambdaName(),
                                                       getInfo());

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

      // return the generated tcap
      return std::move(tcapString);
    }

    /**
     * 1. Check if this is already joined and filtered if it is, we don't need to do anything
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
     * 2. This is a predicate but it is not joined therefore we need to do a cartasian join there is no need to do a
     * filter after this since the inputs are already filtered and this is an and predicate.
     */


    /**
     * 2.1. Create a hash one for the LHS side
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
    mustache::mustache leftOutputTupleTemplate{"{{tupleSetMidTag}}_hashOneFor{{LHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}"};
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
     * 2.2 Create a hash one for the RHS side
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
    mustache::mustache rightOutputTupleSetNameTemplate{"{{tupleSetMidTag}}_hashOneFor{{RHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}"};
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
     * 2.3 Make the cartasian join
     */

    mustache::mustache outputTupleSetTemplate{"{{tupleSetMidTag}}_CartesianJoined{{computationLabel}}_{{lambdaLabel}}"};
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

    // return the string
    return std::move(tcapString);
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