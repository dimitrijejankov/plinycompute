/*****************************************************************************
 *                                 *
 *  Copyright 2018 Rice University                   *
 *                                 *
 *  Licensed under the Apache License, Version 2.0 (the "License");    *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                *
 *                                 *
 *      http://www.apache.org/licenses/LICENSE-2.0               *
 *                                 *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                   *
 *                                 *
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
#include "mustache.h"

namespace pdb {

template <class Out, class ClassType>
class AttAccessLambda : public TypedLambdaObject <Ptr <Out>> {

public:

  // create an att access lambda; offset is the position in the input object where we are going to find the input att
  AttAccessLambda (std::string inputTypeNameIn, std::string attNameIn, std::string attType, Handle <ClassType> & input, size_t offset) :
             offsetOfAttToProcess (offset), inputTypeName (std::move(inputTypeNameIn)), attName (std::move(attNameIn)), attTypeName (std::move(attType)) {
    this->setInputIndex(0, -(input.getExactTypeInfoValue() + 1));
  }

  ComputeExecutorPtr getExecutor (TupleSpec &inputSchema, TupleSpec &attsToOperateOn, TupleSpec &attsToIncludeInOutput) override {

    // create the output tuple set
    TupleSetPtr output = std::make_shared <TupleSet> ();

    // create the machine that is going to setup the output tuple set, using the input tuple set
    TupleSetSetupMachinePtr myMachine = std::make_shared <TupleSetSetupMachine> (inputSchema, attsToIncludeInOutput);

    // this is the input attribute that we will process
    std::vector <int> matches = myMachine->match (attsToOperateOn);
    int whichAtt = matches[0];

    // this is the output attribute
    int outAtt = attsToIncludeInOutput.getAtts ().size ();

    return std::make_shared <ApplyComputeExecutor> (output, [=] (TupleSetPtr input) {

      // set up the output tuple set
      myMachine->setup (input, output);

      // get the columns to operate on
      std::vector <Handle<ClassType>> &inputColumn = input->getColumn <Handle<ClassType>> (whichAtt);

      // setup the output column, if it is not already set up
      if (!output->hasColumn (outAtt)) {
        output->addColumn (outAtt, new std::vector <Ptr <Out>>, true);
      }

      // get the output column
      std::vector <Ptr <Out>> &outColumn = output->getColumn <Ptr <Out>> (outAtt);

      // loop down the columns, setting the output
      int numTuples = inputColumn.size ();
      outColumn.resize (numTuples);

      for (int i = 0; i < numTuples; i++) {

        auto ptr = (char *) &(*(inputColumn[i]));
        auto offset = offsetOfAttToProcess;
        auto t = (Out *) (ptr + offset);
        outColumn [i] = t;
      }

      return output;
    });
  }

  std::string getTypeOfLambda () const override {
    return std::string ("attAccess");
  }

  std::string typeOfAtt () {
    return attTypeName;
  }

  std::string whichAttWeProcess () {
    return attName;
  }

  std::string getInputType () {
    return inputTypeName;
  }

  unsigned int getNumInputs() override {
      return 1;
  }

  std::map<std::string, std::string> getInfo() override {

    // fill in the info
    return std::map<std::string, std::string> {
      std::make_pair ("lambdaType", getTypeOfLambda()),
      std::make_pair ("inputTypeName", inputTypeName),
      std::make_pair ("attName", attName),
      std::make_pair ("attTypeName", attTypeName)
    };
  };

private:

  size_t offsetOfAttToProcess;
  std::string inputTypeName;
  std::string attName;
  std::string attTypeName;
};

}