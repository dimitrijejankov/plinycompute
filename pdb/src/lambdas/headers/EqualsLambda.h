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
#include "ComputeExecutor.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include "Ptr.h"
#include "PDBMap.h"
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

private:

  LambdaTree<LeftType> lhs;
  LambdaTree<RightType> rhs;
};

}

#endif
