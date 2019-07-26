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

#ifndef DEREF_LAM_H
#define DEREF_LAM_H

#include <string>
#include <utility>
#include <vector>

#include "Ptr.h"
#include "Handle.h"
#include "TupleSet.h"
#include "executors/ApplyComputeExecutor.h"
#include "TupleSetMachine.h"
#include "LambdaTree.h"
#include "MultiInputsBase.h"
#include "TypedLambdaObject.h"
#include "mustache.h"

namespace pdb {

template<class OutType>
class DereferenceLambda : public TypedLambdaObject<OutType> {

public:

  explicit DereferenceLambda(LambdaTree<Ptr<OutType>> &input) : input(input) {}

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

    // this is the output attribute
    int outAtt = (int) attsToIncludeInOutput.getAtts().size();

    return std::make_shared<ApplyComputeExecutor>(
        output,
        [=](TupleSetPtr input) {

          // set up the output tuple set
          myMachine->setup(input, output);

          // get the columns to operate on
          std::vector<Ptr<OutType>> &inColumn = input->getColumn<Ptr<OutType>>(firstAtt);

          // create the output attribute, if needed
          if (!output->hasColumn(outAtt)) {
            output->addColumn(outAtt, new std::vector<OutType>, true);
          }

          // get the output column
          std::vector<OutType> &outColumn = output->getColumn<OutType>(outAtt);

          // loop down the columns, setting the output
          auto numTuples = inColumn.size();
          outColumn.resize(numTuples);
          for (int i = 0; i < numTuples; i++) {
            outColumn[i] = *inColumn[i];
          }
          return output;
        }
    );

  }

  std::string getTypeOfLambda() override {
    return std::string("deref");
  }

  unsigned int getNumInputs() override {
    return 1;
  }

  int getNumChildren() override {
    return 1;
  }

  LambdaObjectPtr getChild(int which) override {
    if (which == 0) {
      return input.getPtr();
    }

    return nullptr;
  }

  std::string getTCAPString(const std::string &inputTupleSetName,
                            const std::vector<std::string> &inputColumnNames,
                            const std::vector<std::string> &inputColumnsToApply,
                            const std::string &outputTupleSetName,
                            const std::vector<std::string> &outputColumns,
                            const std::string &outputColumnName,
                            const std::string &tcapOperation,
                            const std::string &computationNameAndLabel,
                            const std::string &lambdaNameAndLabel,
                            const std::map<std::string, std::string> &info) {

    mustache::mustache outputTupleSetNameTemplate
        {"{{outputTupleSetName}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}) <= "
         "{{tcapOperation}} ({{inputTupleSetName}}({{#inputColumnsToApply}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnsToApply}}), "
         "{{inputTupleSetName}}({{#hasColumnNames}}{{#inputColumnNames}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnNames}}{{/hasColumnNames}}), "
         "'{{computationNameAndLabel}}', "
         "{{#hasLambdaNameAndLabel}}'{{lambdaNameAndLabel}}', {{/hasLambdaNameAndLabel}}"
         "[{{#info}}('{{key}}', '{{value}}'){{^isLast}}, {{/isLast}}{{/info}}])\n"};

    // create the data for the output columns
    mustache::data outputColumnData = mustache::from_vector<std::string>(outputColumns);

    // create the data for the input columns to apply
    mustache::data inputColumnsToApplyData = mustache::from_vector<std::string>(inputColumnsToApply);

    // create the data for the input columns to apply
    mustache::data inputColumnNamesData = mustache::from_vector<std::string>(inputColumnNames);

    // create the info data
    mustache::data infoData = mustache::from_map(info);

    // create the data for the lambda
    mustache::data lambdaData;

    lambdaData.set("outputTupleSetName", outputTupleSetName);
    lambdaData.set("outputColumns", outputColumnData);
    lambdaData.set("tcapOperation", tcapOperation);
    lambdaData.set("inputTupleSetName", inputTupleSetName);
    lambdaData.set("inputColumnsToApply", inputColumnsToApplyData);
    lambdaData.set("hasColumnNames", !inputColumnNames.empty());
    lambdaData.set("inputColumnNames", inputColumnNamesData);
    lambdaData.set("inputTupleSetName", inputTupleSetName);
    lambdaData.set("computationNameAndLabel", computationNameAndLabel);
    lambdaData.set("hasLambdaNameAndLabel", !lambdaNameAndLabel.empty());
    lambdaData.set("lambdaNameAndLabel", lambdaNameAndLabel);
    lambdaData.set("info", infoData);

    return outputTupleSetNameTemplate.render(lambdaData);
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
                           std::string &lambdaName,
                           MultiInputsBase *multiInputsComp = nullptr,
                           bool amIPartOfJoinPredicate = false,
                           bool amILeftChildOfEqualLambda = false,
                           bool amIRightChildOfEqualLambda = false,
                           std::string parentLambdaName = "",
                           bool isSelfJoin = false) override {

    // create the data for the lambda
    mustache::data lambdaData;
    lambdaData.set("typeOfLambda", getTypeOfLambda());
    lambdaData.set("lambdaLabel", std::to_string(lambdaLabel));
    lambdaData.set("computationName", computationName);
    lambdaData.set("computationLabel", std::to_string(computationLabel));

    // create the computation name with label
    mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
    std::string computationNameWithLabel = computationNameWithLabelTemplate.render(lambdaData);

    // create the lambda name
    mustache::mustache lambdaNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}"};
    lambdaName = lambdaNameTemplate.render(lambdaData);

    std::string inputTupleSetName;
    std::string tupleSetMidTag;
    int index;

    if (multiInputsComp == nullptr) {
      tupleSetMidTag = "OutFor";
      inputTupleSetName = inputTupleSetNames[0];
    } else {
      tupleSetMidTag = "ExtractedFor";
      index = this->getInputIndex(0);
      inputTupleSetName = multiInputsComp->getTupleSetNameForIthInput(index);
      inputColumnNames = multiInputsComp->getInputColumnsForIthInput(index);
      inputColumnsToApply = multiInputsComp->getInputColumnsToApplyForIthInput(index);
    }

    // create the data for the lambda
    lambdaData.set("tupleSetMidTag", tupleSetMidTag);

    // create the output tuple set name
    mustache::mustache outputTupleSetNameTemplate{"deref_{{lambdaLabel}}{{tupleSetMidTag}}{{computationName}}{{computationLabel}}"};
    outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // create the output column name
      mustache::mustache outputColumnNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}_{{computationLabel}}_{{tupleSetMidTag}}"};
      outputColumnName = outputColumnNameTemplate.render(lambdaData);

    // fill up the output columns and setup the data
    outputColumns.clear();
    for (const auto &inputColumnName : inputColumnNames) {
      if (inputColumnName != inputColumnsToApply[0]) {
        outputColumns.push_back(inputColumnName);
      }
    }
    outputColumns.push_back(outputColumnName);

    // form the input columns to keep
    std::vector<std::string> inputColumnsToKeep;
    for (const auto &inputColumnName : inputColumnNames) {
      if (std::find(inputColumnsToApply.begin(), inputColumnsToApply.end(), inputColumnName) == inputColumnsToApply.end()) {
        // add the data
        inputColumnsToKeep.push_back(inputColumnName);
      }
    }

    // the tcap string
    std::string tcapString = formatAtomicComputation(inputTupleSetName,
                                                     inputColumnsToKeep,
                                                     inputColumnsToApply,
                                                     outputTupleSetName,
                                                     outputColumns,
                                                     outputColumnName,
                                                     "APPLY",
                                                     computationNameWithLabel,
                                                     lambdaName,
                                                     getInfo());

    if (multiInputsComp == nullptr) {
      return tcapString;
    }

    if (amILeftChildOfEqualLambda || amIRightChildOfEqualLambda) {
      inputTupleSetName = outputTupleSetName;
      inputColumnNames.clear();
      for (const auto &outputColumn : outputColumns) {
        // we want to remove the extracted value column from here
        if (outputColumn != outputColumnName) {
          inputColumnNames.push_back(outputColumn);
        }
      }
      inputColumnsToApply.clear();
      inputColumnsToApply.push_back(outputColumnName);

      std::string hashOperator = amILeftChildOfEqualLambda ? "HASHLEFT" : "HASHRIGHT";
      outputTupleSetName = outputTupleSetName.append("_hashed");
      outputColumnName = outputColumnName.append("_hash");
      outputColumns.clear();

      for (const auto &inputColumnName : inputColumnNames) {
        outputColumns.push_back(inputColumnName);
      }
      outputColumns.push_back(outputColumnName);

      tcapString += this->getTCAPString(inputTupleSetName,
                                        inputColumnNames,
                                        inputColumnsToApply,
                                        outputTupleSetName,
                                        outputColumns,
                                        outputColumnName,
                                        hashOperator,
                                        computationNameWithLabel,
                                        parentLambdaName,
                                        std::map<std::string, std::string>());
    }

    if (!isSelfJoin) {
      for (unsigned int i = 0; i < multiInputsComp->getNumInputs(); i++) {
        std::string curInput = multiInputsComp->getNameForIthInput(i);
        auto iter = std::find(outputColumns.begin(), outputColumns.end(), curInput);
        if (iter != outputColumns.end()) {
          multiInputsComp->setTupleSetNameForIthInput(i, outputTupleSetName);
          multiInputsComp->setInputColumnsForIthInput(i, outputColumns);
          multiInputsComp->addColumnToInputColumnsToApplyForIthInput(i, outputColumnName);
        }
      }
    } else {
      // only update myIndex
      multiInputsComp->setTupleSetNameForIthInput(index, outputTupleSetName);
      multiInputsComp->setInputColumnsForIthInput(index, outputColumns);
      multiInputsComp->addColumnToInputColumnsToApplyForIthInput(index, outputColumnName);
    }

    return tcapString;
  }

  std::map<std::string, std::string> getInfo() override {

    // fill in the info
    return std::map<std::string, std::string>{
        std::make_pair("lambdaType", getTypeOfLambda()),
    };
  };

 private:

  LambdaTree<Ptr<OutType>> input;
};

}

#endif
