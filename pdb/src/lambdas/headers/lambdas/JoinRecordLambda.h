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

template<class OutType, class LeftType, class RightType>
class JoinRecordLambda : public TypedLambdaObject<Handle<OutType>> {
public:

  JoinRecordLambda(LambdaObjectPtr keyLambda, LambdaObjectPtr valueLambda) {

    // add the children
    this->children[0] = keyLambda;
    this->children[1] = valueLambda;
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
    int keyAtt = inputAtts[0];
    int valueAtt = inputAtts[1];

    // this is the output attribute
    auto outAtt = (int) attsToIncludeInOutput.getAtts().size();

    return std::make_shared<ApplyComputeExecutor>(
        output,
        [=](TupleSetPtr input) {

          // set up the output tuple set
          myMachine->setup(input, output);

          // get the columns to operate on
          std::vector<Handle<LeftType>>& keyColumn = input->getColumn<Handle<LeftType>>(keyAtt);
          std::vector<Handle<RightType>>& valueColumn = input->getColumn<Handle<RightType>>(valueAtt);

          // create the output attribute, if needed
          if (!output->hasColumn(outAtt)) {
            auto outColumn = new std::vector<Handle<OutType>>;
            output->addColumn(outAtt, outColumn, true);
          }

          // get the output column
          std::vector<Handle<OutType>>& outColumn = output->getColumn<Handle<OutType>>(outAtt);

          // loop down the columns, setting the output
          auto numTuples = keyColumn.size();
          outColumn.resize(numTuples);
          for (int i = 0; i < numTuples; i++) {
            outColumn[i] = makeObject<OutType>();
            outColumn[i]->getKey() = *keyColumn[i];
            outColumn[i]->getValue() = *valueColumn[i];
          }
          return output;
        });
  }

  std::string getTypeOfLambda() const override {
    return std::string("joinRec");
  }

  unsigned int getNumInputs() override {
    return 2;
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