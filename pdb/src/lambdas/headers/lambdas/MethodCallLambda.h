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

#ifndef METHOD_CALL_LAM_H
#define METHOD_CALL_LAM_H

#include <vector>
#include "Lambda.h"
#include "executors/ComputeExecutor.h"

namespace pdb {

template<class Out, class ClassType>
class MethodCallLambda : public TypedLambdaObject<Out> {

 public:

  std::function<ComputeExecutorPtr(TupleSpec & , TupleSpec & , TupleSpec & )> getExecutorFunc;
  std::function<bool(std::string &, TupleSetPtr, int)> columnBuilder;
  std::string inputTypeName;
  std::string methodName;
  std::string returnTypeName;

 public:

  // create an att access lambda; offset is the position in the input object where we are going to find the input att
  MethodCallLambda(std::string inputTypeName,
                   std::string methodName,
                   std::string returnTypeName,
                   Handle<ClassType> &input,
                   std::function<bool(std::string &, TupleSetPtr, int)> columnBuilder,
                   std::function<ComputeExecutorPtr(TupleSpec & , TupleSpec & , TupleSpec & )> getExecutorFunc) :
      getExecutorFunc(std::move(getExecutorFunc)), columnBuilder(std::move(columnBuilder)), inputTypeName(std::move(inputTypeName)),
      methodName(std::move(methodName)), returnTypeName(std::move(returnTypeName)) {

    std::cout << "MethodCallLambda: input type code is " << input.getExactTypeInfoValue() << std::endl;

  }

  std::string getTypeOfLambda() override {
    return std::string("methodCall");
  }

  std::string whichMethodWeCall() {
    return methodName;
  }

  std::string getInputType() {
    return inputTypeName;
  }

  std::string getOutputType() override {
    return returnTypeName;
  }

  int getNumChildren() override {
    return 0;
  }

  LambdaObjectPtr getChild(int which) override {
    return nullptr;
  }

  ComputeExecutorPtr getExecutor(TupleSpec &inputSchema,
                                 TupleSpec &attsToOperateOn,
                                 TupleSpec &attsToIncludeInOutput) override {
    return getExecutorFunc(inputSchema, attsToOperateOn, attsToIncludeInOutput);
  }

  unsigned int getNumInputs() override {
    return 1;
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
                           std::string &lambdaName,
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

    // create the computation name with label
    mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
    std::string computationNameWithLabel = computationNameWithLabelTemplate.render(lambdaData);

    // create the lambda name
    mustache::mustache lambdaNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}"};
    lambdaName = lambdaNameTemplate.render(lambdaData);

    // things we need to figure out in the next step
    int myIndex = -1;
    std::string inputTupleSetName;
    std::string tupleSetMidTag;
    std::string originalInputColumnToApply;

    if (multiInputsComp == nullptr) {
      tupleSetMidTag = "OutFor_";
      inputTupleSetName = inputTupleSetNames[0];
    } else {
      tupleSetMidTag = "ExtractedFor_";
      myIndex = this->getInputIndex(0);
      PDB_COUT << lambdaName << ": myIndex=" << myIndex << std::endl;
      inputTupleSetName = multiInputsComp->getTupleSetNameForIthInput(myIndex);
      PDB_COUT << "inputTupleSetName=" << inputTupleSetName << std::endl;
      inputColumnNames = multiInputsComp->getInputColumnsForIthInput(myIndex);
      inputColumnsToApply.clear();
      inputColumnsToApply.push_back(multiInputsComp->getNameForIthInput(myIndex));
      originalInputColumnToApply = multiInputsComp->getNameForIthInput(myIndex);
      PDB_COUT << "originalInputColumnToApply=" << originalInputColumnToApply << std::endl;
    }

    // set the lambda data
    lambdaData.set("tupleSetMidTag", tupleSetMidTag);
    lambdaData.set("methodName", methodName);

    // create the output tuple set name
    mustache::mustache outputTupleSetNameTemplate
        {"methodCall_{{lambdaLabel}}{{tupleSetMidTag}}{{computationName}}{{computationLabel}}"};
    outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

    // create the output column name
    mustache::mustache outputColumnNameTemplate{"methodCall_{{lambdaLabel}}{{tupleSetMidTag}}_{{methodName}}"};
    outputColumnName = outputColumnNameTemplate.render(lambdaData);

    // initialize the output columns
    outputColumns.clear();
    for (const auto &inputColumnName : inputColumnNames) {
      outputColumns.push_back(inputColumnName);
    }
    outputColumns.push_back(outputColumnName);

    // the additional info about this attribute access lambda
    std::map<std::string, std::string> info;

    // generate the TCAP string for the lambda
    std::string tcapString;
    tcapString += this->getTCAPString(inputTupleSetName,
                                      inputColumnNames,
                                      inputColumnsToApply,
                                      outputTupleSetName,
                                      outputColumns,
                                      outputColumnName,
                                      "APPLY",
                                      computationNameWithLabel,
                                      lambdaName,
                                      getInfo());

    // is a multi input computation just return the tcapString
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
      outputTupleSetName = outputTupleSetName + "_hashed";
      outputColumnName = outputColumnName + "_hash";
      outputColumns.clear();

      std::copy(inputColumnNames.begin(), inputColumnNames.end(), std::back_inserter(outputColumns));
      outputColumns.push_back(outputColumnName);;

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
      for (unsigned int index = 0; index < multiInputsComp->getNumInputs(); index++) {
        std::string curInput = multiInputsComp->getNameForIthInput(index);
        auto iter = std::find(outputColumns.begin(), outputColumns.end(), curInput);
        if (iter != outputColumns.end()) {
          PDB_COUT << "MultiInputBase for index=" << index << " is updated" << std::endl;
          multiInputsComp->setTupleSetNameForIthInput(index, outputTupleSetName);
          multiInputsComp->setInputColumnsForIthInput(index, outputColumns);
          multiInputsComp->setInputColumnsToApplyForIthInput(index, outputColumnName);
        }
        if (originalInputColumnToApply == curInput) {
          PDB_COUT << "MultiInputBase for index=" << index << " is updated" << std::endl;
          multiInputsComp->setTupleSetNameForIthInput(index, outputTupleSetName);
          multiInputsComp->setInputColumnsForIthInput(index, outputColumns);
          multiInputsComp->setInputColumnsToApplyForIthInput(index, outputColumnName);
        }
      }
    } else {
      // only update myIndex, I am a self-join
      multiInputsComp->setTupleSetNameForIthInput(myIndex, outputTupleSetName);
      multiInputsComp->setInputColumnsForIthInput(myIndex, outputColumns);
      multiInputsComp->setInputColumnsToApplyForIthInput(myIndex, outputColumnName);
    }

    return tcapString;
  }
  /**
   * Returns the additional information about this lambda currently lambda type,
   * inputTypeName, methodName and returnTypeName
   * @return the map
   */
  std::map<std::string, std::string> getInfo() override {
    // fill in the info
    return std::map<std::string, std::string>{
        std::make_pair("lambdaType", getTypeOfLambda()),
        std::make_pair("inputTypeName", inputTypeName),
        std::make_pair("methodName", methodName),
        std::make_pair("returnTypeName", returnTypeName)
    };
  };

};

}

#endif
