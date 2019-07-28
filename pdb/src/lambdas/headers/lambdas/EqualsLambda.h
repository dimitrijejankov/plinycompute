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

#ifndef EQUALS_LAM_H
#define EQUALS_LAM_H

#include <vector>
#include "Lambda.h"
#include "executors/ComputeExecutor.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include "Ptr.h"
#include "PDBMap.h"
#include <TypedLambdaObject.h>

namespace pdb {

// only one of these two versions is going to work... used to automatically hash on the underlying type
// in the case of a Ptr<> type
template<class MyType>
std::enable_if_t<std::is_base_of<PtrBase, MyType>::value, size_t> hashHim(MyType &him) {
  return Hasher<decltype(*him)>::hash(*him);
}

template<class MyType>
std::enable_if_t<!std::is_base_of<PtrBase, MyType>::value, size_t> hashHim(MyType &him) {
  return Hasher<MyType>::hash(him);
}

// only one of these four versions is going to work... used to automatically dereference a Ptr<blah>
// type on either the LHS or RHS of an equality check
template<class LHS, class RHS>
std::enable_if_t<std::is_base_of<PtrBase, LHS>::value && std::is_base_of<PtrBase, RHS>::value,
                 bool> checkEquals(LHS &lhs, RHS &rhs) {
  return *lhs == *rhs;
}

template<class LHS, class RHS>
std::enable_if_t<std::is_base_of<PtrBase, LHS>::value && !(std::is_base_of<PtrBase, RHS>::value),
                 bool> checkEquals(LHS &lhs, RHS &rhs) {
  return *lhs == rhs;
}

template<class LHS, class RHS>
std::enable_if_t<!(std::is_base_of<PtrBase, LHS>::value) && std::is_base_of<PtrBase, RHS>::value,
                 bool> checkEquals(LHS &lhs, RHS &rhs) {
  return lhs == *rhs;
}

template<class LHS, class RHS>
std::enable_if_t<!(std::is_base_of<PtrBase, LHS>::value) && !(std::is_base_of<PtrBase, RHS>::value), bool> checkEquals(
    LHS &lhs,
    RHS &rhs) {
  return lhs == rhs;
}

template<class LeftType, class RightType>
class EqualsLambda : public TypedLambdaObject<bool> {

public:

  EqualsLambda(LambdaTree<LeftType> lhsIn, LambdaTree<RightType> rhsIn) {
    lhs = lhsIn;
    rhs = rhsIn;
    this->setInputIndex(0, lhs.getInputIndex(0));
    this->setInputIndex(1, rhs.getInputIndex(0));
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

  std::string getTypeOfLambda() override {
    return std::string("==");
  }

  int getNumChildren() override {
    return 2;
  }

  LambdaObjectPtr getChild(int which) override {
    if (which == 0)
      return lhs.getPtr();
    if (which == 1)
      return rhs.getPtr();
    return nullptr;
  }

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

    std::string tcapString;
    if (multiInputsComp == nullptr) {

      /**
       * 1. This is not a join where the equals lambda is applied therefore we simply need to generate a atomic computation
       *    for the boolean lambda
       */

      std::string inputTupleSetName = inputTupleSetNames[0];

      // create the output tuple set name
      mustache::mustache outputTupleSetNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}{{tupleSetMidTag}}{{computationName}}{{computationLabel}}"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // create the output column name
      mustache::mustache outputColumnNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}_{{computationLabel}}_{{tupleSetMidTag}}"};
      outputColumnName = outputColumnNameTemplate.render(lambdaData);

      // forward the input columns to the output and add the new output column to it
      outputColumns = inputColumnNames;
      outputColumns.push_back(outputColumnName);

      tcapString += formatLambdaComputation(inputTupleSetName,
                                            inputColumnNames,
                                            inputColumnsToApply,
                                            outputTupleSetName,
                                            outputColumns,
                                            outputColumnName,
                                            "APPLY",
                                            computationNameWithLabel,
                                            myLambdaName,
                                            getInfo());

    } else {

      /**
       * 1.1 Form the left hasher
       */

      // grab the lhs indices
      auto lhsInput = getInputIndex(0);

      // the name of the lhs input tuple set
      auto lhsInputTupleSet = multiInputsComp->getTupleSetNameForIthInput(lhsInput);

      // the input columns that we are going to forward
      auto lhsInputColumns = multiInputsComp->getNotAppliedInputColumnsForIthInput(lhsInput);

      // the input to the hash can only be one column
      auto lhsInputColumnsToApply = multiInputsComp->getInputColumnsToApplyForIthInput(lhsInput);
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
                                            lhsOutputColumnName,
                                            "HASHLEFT",
                                            computationNameWithLabel,
                                            myLambdaName,
                                            {});

      /**
       * 1.2 Form the left hasher
       */

      auto rhsInput = getInputIndex(1);

      // the name of the lhs input tuple set
      auto rhsInputTupleSet = multiInputsComp->getTupleSetNameForIthInput(rhsInput);

      // the input columns that we are going to forward
      auto rhsInputColumns = multiInputsComp->getNotAppliedInputColumnsForIthInput(rhsInput);

      // the input to the hash can only be one column
      auto rhsInputColumnsToApply = multiInputsComp->getInputColumnsToApplyForIthInput(rhsInput);
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
                                            rhsOutputColumnName,
                                            "HASHRIGHT",
                                            computationNameWithLabel,
                                            myLambdaName,
                                            {});


      /**
       * 2. First we form a join computation that joins based on the hash columns
       */

      outputTupleSetName = "JoinedFor_equals" + std::to_string(lambdaLabel) + computationName + std::to_string(computationLabel);

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
       * 3.1 Next we extract the LHS column of the join from the lhs input
       */

      // the output of the join is the input to the lambda that extracts the LHS
      std::string inputTupleSetName = outputTupleSetName;

      // the input to the lambda is the output
      inputColumnNames = outputColumns;

      // get the column that the child is applying
      inputColumnsToApply = { multiInputsComp->getNameForIthInput(lhs.getInputIndex(0)) };

      // make the name for the column we are going to create
      mustache::mustache outputColumnNameTemplate{"LHSExtractedFor_{{lambdaLabel}}_{{computationLabel}}"};
      std::string lhsColumn = outputColumnNameTemplate.render(lambdaData);
      outputColumns.push_back(lhsColumn);

      // make the name for tuple set created
      mustache::mustache outputTupleSetNameTemplate{"{{tupleSetNamePrefix}}_WithLHSExtracted"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // the additional info about this attribute access lambda
      tcapString += formatLambdaComputation(inputTupleSetName,
                                            inputColumnNames,
                                            inputColumnsToApply,
                                            outputTupleSetName,
                                            outputColumns,
                                            outputColumnName,
                                            "APPLY",
                                            computationNameWithLabel,
                                            childrenLambdaNames[0],
                                            getChild(0)->getInfo());

      /**
       * 3.2 Next we extract the RHS column of the join from the rhs input
       */

      // the input to RHS extraction is the output of the extracted LHS
      inputTupleSetName = outputTupleSetName;

      // same goes for the output columns
      inputColumnNames = outputColumns;

      // get the column that the child is applying to get the RHS
      inputColumnsToApply = { multiInputsComp->getNameForIthInput(rhs.getInputIndex(0)) };

      // make the name for tuple set created
      outputTupleSetNameTemplate = {"{{tupleSetNamePrefix}}_WithBOTHExtracted"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // make the name for the column we are going to create
      outputColumnNameTemplate = {"RHSExtractedFor_{{lambdaLabel}}_{{computationLabel}}"};
      std::string rhsColumn = outputColumnNameTemplate.render(lambdaData);
      outputColumns.push_back(rhsColumn);

      // add the tcap string
      tcapString += formatLambdaComputation(inputTupleSetName,
                                            inputColumnNames,
                                            inputColumnsToApply,
                                            outputTupleSetName,
                                            outputColumns,
                                            outputColumnName,
                                            "APPLY",
                                            computationNameWithLabel,
                                            childrenLambdaNames[1],
                                            getChild(1)->getInfo());

      /**
       * 4. Now with both sides extracted we perform the boolean expression that check if we should keep the joined rows
       */

      // the input to the boolean lambda is the output tuple set from the previous lambda
      inputTupleSetName = outputTupleSetName;

      // the boolean lambda is applied on the lhs and rhs extracted column
      inputColumnsToApply = { lhsColumn , rhsColumn };

      // input columns are basically the input columns that are not the hash from the lhs and rhs side
      inputColumnNames = lhsInputColumns;
      inputColumnNames.insert(inputColumnNames.end(), rhsInputColumns.begin(), rhsInputColumns.end());

      // the output columns are the input columns with the additional new column we created to store the comparison result
      outputColumns = inputColumnNames;
      outputColumnNameTemplate = {"bool_{{lambdaLabel}}_{{computationLabel}}"};
      std::string booleanColumn = outputColumnNameTemplate.render(lambdaData);
      outputColumns.emplace_back(booleanColumn);

      // we make a new tupleset name
      outputTupleSetNameTemplate = {"{{tupleSetNamePrefix}}_BOOL"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // make the comparison lambda
      tcapString += formatLambdaComputation(inputTupleSetName,
                                            inputColumnNames,
                                            inputColumnsToApply,
                                            outputTupleSetName,
                                            outputColumns,
                                            outputColumnName,
                                            "APPLY",
                                            computationNameWithLabel,
                                            myLambdaName,
                                            getInfo());

      /**
       * 5. With the boolean expression we preform a filter, that is going to remove all the ones that are false.
       */

      // the previous lambda that created the boolean column is the input to the filter
      inputTupleSetName = outputTupleSetName;

      // this basically removes "_BOOL" column from the output columns since we are done with it after the filter
      outputColumns.pop_back();


      // make the name for the new tuple set that is the output of the filter
      outputTupleSetNameTemplate = {"{{tupleSetNamePrefix}}_FILTERED"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // make the filter
      tcapString += formatFilterComputation(outputTupleSetName,
                                            outputColumns,
                                            inputTupleSetName,
                                            { booleanColumn },
                                            inputColumnNames,
                                            computationNameWithLabel);

      // there is no output column here for some reason TODO figure this out
      outputColumnName = "";

      if (!isSelfJoin) {

        // update all the inputs
        for (unsigned int index = 0; index < multiInputsComp->getNumInputs(); index++) {
          std::string curInput = multiInputsComp->getNameForIthInput(index);
          auto iter = std::find(outputColumns.begin(), outputColumns.end(), curInput);
          if (iter != outputColumns.end()) {
            multiInputsComp->setTupleSetNameForIthInput(index, outputTupleSetName);
            multiInputsComp->setInputColumnsForIthInput(index, outputColumns);
            multiInputsComp->setColumnToApplyForIthInput(index, outputColumnName);
          }
        }
      }
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

private:

  LambdaTree<LeftType> lhs;
  LambdaTree<RightType> rhs;
};

}

#endif
