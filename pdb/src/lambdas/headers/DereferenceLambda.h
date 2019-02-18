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

#include <vector>
#include "LambdaHelperClasses.h"
#include "ComputeExecutor.h"
#include "ApplyComputeExecutor.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include "Ptr.h"

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


  int getNumChildren() override {
    return 1;
  }

  GenericLambdaObjectPtr getChild(int which) override {
    if (which == 0) {
      return input.getPtr();
    }

    return nullptr;
  }

  std::string toTCAPString(std::string inputTupleSetName,
                           std::vector<std::string> inputColumnNames,
                           std::vector<std::string> inputColumnsToApply,
                           int lambdaLabel,
                           std::string computationName,
                           int computationLabel,
                           std::string &outputTupleSetName,
                           std::vector<std::string> &outputColumns,
                           std::string &outputColumnName) override {
    std::string tcapString;
    outputTupleSetName = "deref_" + std::to_string(lambdaLabel) + "OutFor" + computationName + std::to_string(computationLabel);

    outputColumnName = inputColumnsToApply[0];
    PDB_COUT << "OuputColumnName: " << outputColumnName << std::endl;
    outputColumns.clear();
    for (const auto &inputColumnName : inputColumnNames) {
      outputColumns.push_back(inputColumnName);
    }
    tcapString += outputTupleSetName + "(" + outputColumns[0];
    for (int i = 1; i < outputColumns.size(); i++) {
      tcapString += ",";
      tcapString += outputColumns[i];
    }
    tcapString += ") <= APPLY (";
    tcapString += inputTupleSetName + "(" + inputColumnsToApply[0];
    for (int i = 1; i < inputColumnsToApply.size(); i++) {
      tcapString += ",";
      tcapString += inputColumnsToApply[i];
    }
    std::vector<std::string> inputColumnsToKeep;
    for (const auto &inputColumnName : inputColumnNames) {
      int j = 0;
      for (j = 0; j < inputColumnsToApply.size(); j++) {
        if (inputColumnName == inputColumnsToApply[j]) {
          break;
        }
      }
      if (j == inputColumnsToApply.size()) {
        inputColumnsToKeep.push_back(inputColumnName);
      }
    }
    tcapString += "), " + inputTupleSetName + "(" + inputColumnsToKeep[0];
    for (int i = 1; i < inputColumnsToKeep.size(); i++) {
      tcapString += ",";
      tcapString += inputColumnsToKeep[i];
    }
    tcapString += "), '" + computationName + "_" + std::to_string(computationLabel) + "', '" + getTypeOfLambda() + "_"
        + std::to_string(lambdaLabel) + "')\n";
    return tcapString;

  }


private:

  LambdaTree<Ptr<OutType>> input;
};

}

#endif
