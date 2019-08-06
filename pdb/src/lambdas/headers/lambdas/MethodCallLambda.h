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

#pragma once

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

    this->setInputIndex(0, -(input.getExactTypeInfoValue() + 1));
  }

  std::string getTypeOfLambda() const override {
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

  ComputeExecutorPtr getExecutor(TupleSpec &inputSchema,
                                 TupleSpec &attsToOperateOn,
                                 TupleSpec &attsToIncludeInOutput) override {
    return getExecutorFunc(inputSchema, attsToOperateOn, attsToIncludeInOutput);
  }

  unsigned int getNumInputs() override {
    return 1;
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