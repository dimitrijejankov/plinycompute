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
#include "TupleSet.h"
#include "Ptr.h"
#include "PDBMap.h"
#include <TypedLambdaObject.h>

namespace pdb {

// only one of these three versions is going to work... used to automatically hash on the underlying type
// in the case of a Ptr<> type
template<class MyType>
std::enable_if_t<std::is_base_of<PtrBase, MyType>::value, size_t> hashHim(const MyType &him) {
  return Hasher<decltype(*him)>::hash(*him);
}

template<class MyType>
std::enable_if_t<!std::is_base_of<PtrBase, MyType>::value, size_t> hashHim(const MyType &him) {
  return Hasher<MyType>::hash(him);
}

// only one of these five versions is going to work... used to automatically dereference a Ptr<blah>
// type on either the LHS or RHS of an equality check
template<class LHS, class RHS>
std::enable_if_t<std::is_base_of<PtrBase, LHS>::value && std::is_base_of<PtrBase, RHS>::value, bool> checkEquals(const LHS &lhs, const RHS &rhs) {
  return *lhs == *rhs;
}

template<class LHS, class RHS>
std::enable_if_t<std::is_base_of<PtrBase, LHS>::value && !(std::is_base_of<PtrBase, RHS>::value), bool> checkEquals(const LHS &lhs, const RHS &rhs) {
  return *lhs == rhs;
}

template<class LHS, class RHS>
std::enable_if_t<!(std::is_base_of<PtrBase, LHS>::value) && std::is_base_of<PtrBase, RHS>::value, bool> checkEquals(const LHS &lhs, const RHS &rhs) {
  return lhs == *rhs;
}

template<class LHS, class RHS>
std::enable_if_t<!(std::is_base_of<PtrBase, LHS>::value) && !(std::is_base_of<PtrBase, RHS>::value), bool> checkEquals(const LHS &lhs, const RHS &rhs) {
  return lhs == rhs;
}

template<class LeftType, class RightType>
class EqualsLambda : public TypedLambdaObject<bool> {

public:

  EqualsLambda(LambdaTree<LeftType> lhsIn, LambdaTree<RightType> rhsIn) {

    // add the children
    children[0] = lhsIn.getPtr();
    children[1] = rhsIn.getPtr();
  }

  ComputeExecutorPtr getExecutor(TupleSpec &inputSchema,
                                 TupleSpec &attsToOperateOn,
                                 TupleSpec &attsToIncludeInOutput) override {

    // create the output tuple set
    TupleSetPtr output = std::make_shared<TupleSet>();

    // create the machine that is going to setup the output tuple set, using the input tuple set
    TupleSetSetupMachinePtr myMachine = std::make_shared<TupleSetSetupMachine>(inputSchema, attsToIncludeInOutput);

    // these are the input attributes that we will process
    std::vector<int> inputAtts = myMachine->match(attsToOperateOn);
    int firstAtt = inputAtts[0];
    int secondAtt = inputAtts[1];

    // this is the output attribute
    int outAtt = (int) attsToIncludeInOutput.getAtts().size();

    return std::make_shared<ApplyComputeExecutor>(
        output,
        [=](TupleSetPtr input) {

          // set up the output tuple set
          myMachine->setup(input, output);

          // get the columns to operate on
          std::vector<LeftType> &leftColumn = input->getColumn<LeftType>(firstAtt);
          std::vector<RightType> &rightColumn = input->getColumn<RightType>(secondAtt);

          // create the output attribute, if needed
          if (!output->hasColumn(outAtt)) {
            output->addColumn(outAtt, new std::vector<bool>, true);
          }

          // get the output column
          std::vector<bool> &outColumn = output->getColumn<bool>(outAtt);

          // loop down the columns, setting the output
          auto numTuples = leftColumn.size();
          outColumn.resize(numTuples);
          for (int i = 0; i < numTuples; i++) {
            outColumn[i] = checkEquals(leftColumn[i], rightColumn[i]);
          }
          return output;
        }
    );
  }

  ComputeExecutorPtr getRightHasher(TupleSpec &inputSchema,
                                    TupleSpec &attsToOperateOn,
                                    TupleSpec &attsToIncludeInOutput) override {

    // create the output tuple set
    TupleSetPtr output = std::make_shared<TupleSet>();

    // create the machine that is going to setup the output tuple set, using the input tuple set
    TupleSetSetupMachinePtr myMachine = std::make_shared<TupleSetSetupMachine>(inputSchema, attsToIncludeInOutput);

    // these are the input attributes that we will process
    std::vector<int> inputAtts = myMachine->match(attsToOperateOn);
    int secondAtt = inputAtts[0];

    // this is the output attribute
    int outAtt = (int) attsToIncludeInOutput.getAtts().size();

    return std::make_shared<ApplyComputeExecutor>(
        output,
        [=](TupleSetPtr input) {

          // set up the output tuple set
          myMachine->setup(input, output);

          // get the columns to operate on
          std::vector<RightType> &rightColumn = input->getColumn<RightType>(secondAtt);

          // create the output attribute, if needed
          if (!output->hasColumn(outAtt)) {
            output->addColumn(outAtt, new std::vector<size_t>, true);
          }

          // get the output column
          std::vector<size_t> &outColumn = output->getColumn<size_t>(outAtt);

          // loop down the columns, setting the output
          auto numTuples = rightColumn.size();
          outColumn.resize(numTuples);
          for (int i = 0; i < numTuples; i++) {
            outColumn[i] = hashHim(rightColumn[i]);
          }
          return output;
        }
    );
  }

  ComputeExecutorPtr getLeftHasher(TupleSpec &inputSchema,
                                   TupleSpec &attsToOperateOn,
                                   TupleSpec &attsToIncludeInOutput) override {

    // create the output tuple set
    TupleSetPtr output = std::make_shared<TupleSet>();

    // create the machine that is going to setup the output tuple set, using the input tuple set
    TupleSetSetupMachinePtr myMachine = std::make_shared<TupleSetSetupMachine>(inputSchema, attsToIncludeInOutput);

    // these are the input attributes that we will process
    std::vector<int> inputAtts = myMachine->match(attsToOperateOn);
    int firstAtt = inputAtts[0];

    // this is the output attribute
    int outAtt = (int) attsToIncludeInOutput.getAtts().size();

    return std::make_shared<ApplyComputeExecutor>(
        output,
        [=](TupleSetPtr input) {

          // set up the output tuple set
          myMachine->setup(input, output);

          // get the columns to operate on
          std::vector<LeftType> &leftColumn = input->getColumn<LeftType>(firstAtt);

          // create the output attribute, if needed
          if (!output->hasColumn(outAtt)) {
            output->addColumn(outAtt, new std::vector<size_t>, true);
          }

          // get the output column
          std::vector<size_t> &outColumn = output->getColumn<size_t>(outAtt);

          // loop down the columns, setting the output
          auto numTuples = leftColumn.size();
          outColumn.resize(numTuples);
          for (int i = 0; i < numTuples; i++) {
            outColumn[i] = hashHim(leftColumn[i]);
          }
          return output;
        }
    );
  }

  std::string getTypeOfLambda() const override {
    return std::string("==");
  }

  /**
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

    /**
     * 0. Check if we need to perform a join to get the value of this lamda
     */

    // get the left and right child
    auto lhs = children[0];
    auto rhs = children[1];

    // get the columns of the lhs and where they are generated
    auto &lhsColumns = lhs->getGeneratedColumns();
    assert(!lhs->joinedInputs.empty());
    auto lhsIndex = *lhs->joinedInputs.begin();
    auto lhsGroup = multiInputsComp->joinGroupForInput[lhsIndex];

    // get the columns of the rhs and where they are generated
    auto &rhsColumns = rhs->getGeneratedColumns();
    assert(!rhs->joinedInputs.empty());
    auto rhsIndex = *rhs->joinedInputs.begin();
    auto rhsGroup = multiInputsComp->joinGroupForInput[rhsIndex];

    std::string tcapString;

    // check if all the columns are in the same tuple set, in that case we apply the equals lambda directly onto that tuple set
    if (lhsGroup == rhsGroup) {

      /**
       * 1. This is not a join where the equals lambda is applied therefore we simply need to generate a atomic computation
       *    for the boolean lambda and a possible filter if this is a join predicate
       */

      // the input tuple set is the lhs tuple set
      std::string inputTupleSetName = multiInputsComp->tupleSetNamesForInputs[lhsIndex];

      // create the output tuple set name
      mustache::mustache outputTupleSetNameTemplate{"equal_{{lambdaLabel}}{{tupleSetMidTag}}{{computationName}}{{computationLabel}}"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // create the output column name
      mustache::mustache outputColumnNameTemplate{"equal_{{lambdaLabel}}_{{computationLabel}}_{{tupleSetMidTag}}"};
      std::string outputColumnName = outputColumnNameTemplate.render(lambdaData);

      // remove the lhs input if it is not the original input column
      auto inputs = multiInputsComp->inputColumnsForInputs[lhsIndex];
      if(std::find(multiInputsComp->inputNames.begin(), multiInputsComp->inputNames.end(), lhsColumns[0]) == multiInputsComp->inputNames.end()) {
        inputs.erase(std::remove(inputs.begin(), inputs.end(), lhsColumns[0]), inputs.end());
      }

      // remove the rhs input if it is not in the original input colum
      if(std::find(multiInputsComp->inputNames.begin(), multiInputsComp->inputNames.end(), rhsColumns[0]) == multiInputsComp->inputNames.end()) {
        inputs.erase(std::remove(inputs.begin(), inputs.end(), rhsColumns[0]), inputs.end());
      }

      // the output are the forwarded inputs with the generated column
      outputColumns = inputs;
      outputColumns.push_back(outputColumnName);

      // the the columns we have applied in this lambda
      appliedColumns = { lhsColumns[0], rhsColumns[0] };

      // generate the boolean lambda
      tcapString += formatLambdaComputation(inputTupleSetName,
                                            inputs,
                                            appliedColumns,
                                            outputTupleSetName,
                                            outputColumns,
                                            "APPLY",
                                            computationNameWithLabel,
                                            getLambdaName(),
                                            getInfo());

      // we are going to be applying the generated boolean column
      generatedColumns = { outputColumnName };

      // if we are a part of the join predicate we need to apply a filter
      if(isPredicate) {

        // mark as filtered
        isFiltered = true;

        // set the input tuple set name
        inputTupleSetName = outputTupleSetName;

        // create the output tuple set name for the filter
        outputTupleSetNameTemplate = {"filtered_out{{lambdaLabel}}{{tupleSetMidTag}}{{computationName}}{{computationLabel}}"};
        outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

        // remove the boolean column since we are using it in the filter
        outputColumns.pop_back();

        // the input columns are the same as the output columns
        auto &inputColumns = outputColumns;

        // make the filter
        tcapString += formatFilterComputation(outputTupleSetName,
                                              outputColumns,
                                              inputTupleSetName,
                                              generatedColumns,
                                              inputColumns,
                                              computationNameWithLabel);

        // clear the columns to apply since we are not generating anything
        std::swap(appliedColumns, generatedColumns);
        generatedColumns.clear();
      }

      // update the join group
      joinGroup = multiInputsComp->joinGroupForInput[lhsIndex];

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

    } else {

      /**
       * 1.1 Form the left hasher
       */

      if(!isPredicate) {

        // so we don't support stuff like (a == b) == (c == d) since this would make us generate
        // a cartasian join for subexpressions (a == b) and (c == d), this can be done though it would just be
        // anoying to write the code to do that
        throw std::runtime_error("We currently do not support a query that complicated!");
      }

      // the name of the lhs input tuple set
      auto &lhsInputTupleSet = multiInputsComp->tupleSetNamesForInputs[lhsIndex];

      // the input columns that we are going to forward (we only keep the real inputs)
      std::vector<std::string> lhsInputColumns;
      for(auto c : multiInputsComp->getNotAppliedInputColumnsForIthInput(lhsIndex)) {
        if(std::find(multiInputsComp->inputNames.begin(), multiInputsComp->inputNames.end(), c) != multiInputsComp->inputNames.end()){
          lhsInputColumns.emplace_back(c);
        }
      }

      // the input to the hash can only be one column
      auto &lhsInputColumnsToApply = lhsColumns;
      assert(lhsInputColumnsToApply.size() == 1);

      // form the output tuple set name
      std::string lhsOutputTupleSetName = lhsInputTupleSet + "_hashed";

      // the hash column
      auto lhsOutputColumnName = lhsInputColumnsToApply[0] + "_hash";

      // add the hashed column
      auto lhsOutputColumns = lhsInputColumns;
      lhsOutputColumns.emplace_back(lhsOutputColumnName);

      // add the tcap string
      tcapString += formatLambdaComputation(lhsInputTupleSet,
                                            lhsInputColumns,
                                            lhsInputColumnsToApply,
                                            lhsOutputTupleSetName,
                                            lhsOutputColumns,
                                            "HASHLEFT",
                                            computationNameWithLabel,
                                            getLambdaName(),
                                            {});

      /**
       * 1.2 Form the right hasher
       */

      // the name of the lhs input tuple set
      auto &rhsInputTupleSet = multiInputsComp->tupleSetNamesForInputs[rhsIndex];

      // the input columns that we are going to forward
      std::vector<std::string> rhsInputColumns;
      for(auto c : multiInputsComp->getNotAppliedInputColumnsForIthInput(rhsIndex)) {
        if(std::find(multiInputsComp->inputNames.begin(), multiInputsComp->inputNames.end(), c) != multiInputsComp->inputNames.end()){
          rhsInputColumns.emplace_back(c);
        }
      }

      // the input to the hash can only be one column
      auto &rhsInputColumnsToApply = rhsColumns;
      assert(rhsInputColumnsToApply.size() == 1);

      // form the output tuple set name
      std::string rhsOutputTupleSetName = rhsInputTupleSet + "_hashed";

      // the hash column
      auto rhsOutputColumnName = rhsInputColumnsToApply[0] + "_hash";

      // add the hashed column
      auto rhsOutputColumns = rhsInputColumns;
      rhsOutputColumns.emplace_back(rhsOutputColumnName);


      // add the tcap string
      tcapString += formatLambdaComputation(rhsInputTupleSet,
                                            rhsInputColumns,
                                            rhsInputColumnsToApply,
                                            rhsOutputTupleSetName,
                                            rhsOutputColumns,
                                            "HASHRIGHT",
                                            computationNameWithLabel,
                                            getLambdaName(),
                                            {});


      /**
       * 2. First we form a join computation that joins based on the hash columns
       */

      outputTupleSetName = myPrefix + "JoinedFor_equals_" + std::to_string(myLambdaLabel) + myComputationName + std::to_string(myComputationLabel);

      // set the prefix
      lambdaData.set("tupleSetNamePrefix", outputTupleSetName);

      // figure out the output columns, so basically everything that does not have the hash from the hashed lhs and rhs
      outputColumns = lhsInputColumns;
      outputColumns.insert(outputColumns.end(), rhsInputColumns.begin(), rhsInputColumns.end());

      // generate the join computation
      tcapString += formatJoinComputation(outputTupleSetName,
                                          outputColumns,
                                          lhsOutputTupleSetName,
                                          { lhsOutputColumnName },
                                          lhsInputColumns,
                                          rhsOutputTupleSetName,
                                          { rhsOutputColumnName },
                                          rhsInputColumns,
                                          computationNameWithLabel);

      /**
       * 3.0 Update the inputs so that we can reapply the lhs and rhs expressions
       */

      // go through each tuple set and update stuff
      for(int i = 0; i < multiInputsComp->tupleSetNamesForInputs.size(); ++i) {

        // check if this tuple set is the same index
        if (multiInputsComp->joinGroupForInput[i] == rhsGroup || multiInputsComp->joinGroupForInput[i] == lhsGroup) {

          // the output tuple set is the new set with these columns
          multiInputsComp->tupleSetNamesForInputs[i] = outputTupleSetName;
          multiInputsComp->inputColumnsForInputs[i] = outputColumns;
          multiInputsComp->inputColumnsToApplyForInputs[i] = {};

          // this input was joined
          joinedInputs.insert(i);

          // update the join group so that rhs has the same group as lhs
          multiInputsComp->joinGroupForInput[i] = lhsGroup;
        }
      }

      /**
       * 3.1 Next we extract the LHS column of the join from the lhs input
       */

      std::vector<std::string> tcapStrings;
      lhs->generateExpressionTCAP("LExtractedFor" + std::to_string(myLambdaLabel), multiInputsComp, tcapStrings);

      /**
       * 3.2 Next we extract the RHS column of the join from the rhs input
       */

      rhs->generateExpressionTCAP("RExtractedFor" + std::to_string(myLambdaLabel), multiInputsComp, tcapStrings);
      for_each(tcapStrings.begin(), tcapStrings.end(), [&](auto &value) {
        tcapString += value;
      });


      /**
       * 4. Now with both sides extracted we perform the boolean expression that check if we should keep the joined rows
       */

      // the input to the boolean lambda is the output tuple set from the previous lambda
      auto inputTupleSetName = multiInputsComp->tupleSetNamesForInputs[*lhs->joinedInputs.begin()];

      // the boolean lambda is applied on the lhs and rhs extracted column
      appliedColumns = { lhs->generatedColumns.front() , rhs->generatedColumns.front() };

      // input columns are basically the input columns that are not the hash from the lhs and rhs side
      auto inputColumnNames = lhsInputColumns;
      inputColumnNames.insert(inputColumnNames.end(), rhsInputColumns.begin(), rhsInputColumns.end());

      // the output columns are the input columns with the additional new column we created to store the comparison result
      outputColumns = inputColumnNames;
      mustache::mustache outputColumnNameTemplate = {"bool_{{lambdaLabel}}_{{computationLabel}}"};
      std::string booleanColumn = outputColumnNameTemplate.render(lambdaData);
      outputColumns.emplace_back(booleanColumn);

      // we make a new tupleset name
      mustache::mustache outputTupleSetNameTemplate = {"{{tupleSetMidTag}}_{{tupleSetNamePrefix}}_BOOL"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // set the generated columns
      generatedColumns = { booleanColumn };

      // make the comparison lambda
      tcapString += formatLambdaComputation(inputTupleSetName,
                                            inputColumnNames,
                                            appliedColumns,
                                            outputTupleSetName,
                                            outputColumns,
                                            "APPLY",
                                            computationNameWithLabel,
                                            getLambdaName(),
                                            getInfo());

      /**
       * 5. With the boolean expression we preform a filter, that is going to remove all the ones that are false.
       */

      // mark as filtered
      isFiltered = true;

      // the previous lambda that created the boolean column is the input to the filter
      inputTupleSetName = outputTupleSetName;

      // this basically removes "_BOOL" column from the output columns since we are done with it after the filter
      outputColumns.pop_back();

      // make the name for the new tuple set that is the output of the filter
      outputTupleSetNameTemplate = {"{{tupleSetMidTag}}_{{tupleSetNamePrefix}}_FILTERED"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // we are applying the filtering on the boolean column
      appliedColumns = { booleanColumn };

      // we are not generating any columns after the filter
      generatedColumns = {};

      // make the filter
      tcapString += formatFilterComputation(outputTupleSetName,
                                            outputColumns,
                                            inputTupleSetName,
                                            appliedColumns,
                                            inputColumnNames,
                                            computationNameWithLabel);

      // go through each tuple set and update stuff
      for(int i = 0; i < multiInputsComp->tupleSetNamesForInputs.size(); ++i) {

        // check if this tuple set is the same index
        if (multiInputsComp->joinGroupForInput[i] == rhsGroup || multiInputsComp->joinGroupForInput[i] == lhsGroup) {

          // the output tuple set is the new set with these columns
          multiInputsComp->tupleSetNamesForInputs[i] = outputTupleSetName;
          multiInputsComp->inputColumnsForInputs[i] = outputColumns;
          multiInputsComp->inputColumnsToApplyForInputs[i] = generatedColumns;

          // this input was joined
          joinedInputs.insert(i);

          // update the join group so that rhs has the same group as lhs
          multiInputsComp->joinGroupForInput[i] = lhsGroup;
        }
      }

      // update the join group
      joinGroup = multiInputsComp->joinGroupForInput[lhsIndex];
    }

    return tcapString;
  }

  unsigned int getNumInputs() override {
    return 2;
  }

private:

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