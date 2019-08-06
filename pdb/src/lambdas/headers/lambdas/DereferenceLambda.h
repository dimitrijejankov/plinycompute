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

  explicit DereferenceLambda(LambdaTree<Ptr<OutType>> &input) {

    // insert the child
    this->children[0] = input.getPtr();
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

  std::string getTypeOfLambda() const override {
    return std::string("deref");
  }

  unsigned int getNumInputs() override {
    return 1;
  }

  std::map<std::string, std::string> getInfo() override {

    // fill in the info
    return std::map<std::string, std::string>{
        std::make_pair("lambdaType", getTypeOfLambda()),
    };
  };
};

}