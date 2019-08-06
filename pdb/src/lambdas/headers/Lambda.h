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

#ifndef LAMBDA_H
#define LAMBDA_H

#include <memory>
#include <vector>
#include <functional>
#include "Object.h"
#include "Handle.h"
#include "Ptr.h"
#include "TupleSpec.h"
#include "ComputeExecutor.h"
#include "LambdaTree.h"
#include <LambdaObject.h>
#include <DereferenceLambda.h>


namespace pdb {

template<class ReturnType>
class Lambda {

public:

  // in case we wrap up a non-pointer type
  std::shared_ptr<TypedLambdaObject<ReturnType>> tree;

  // is labeled
  bool isLabeled = false;

  // does the actual tree traversal
  static void extractLambdas(std::map<std::string, LambdaObjectPtr> &fillMe,
                             const LambdaObjectPtr& root) {

    // set the lambda
    fillMe[root->getLambdaName()] = root;

    // traverse the child lambdas
    for (const auto &child : root->children) {
      extractLambdas(fillMe, child.second);
    }
  }

public:

  // create a lambda tree that returns a pointer
  Lambda(LambdaTree<Ptr<ReturnType>> treeWithPointer) {

    // a problem is that consumers of this lambda will not be able to deal with a Ptr<ReturnType>...
    // so we need to add an additional operation that dereferences the pointer
    std::shared_ptr<DereferenceLambda<ReturnType>> newRoot = std::make_shared<DereferenceLambda<ReturnType>>(treeWithPointer);
    tree = newRoot;
  }

  // create a lambda tree that returns a non-pointer
  Lambda(LambdaTree<ReturnType> tree) : tree(tree.getPtr()) {}

  template<class LambdaTree>
  void inject(int32_t inputIndex, LambdaTree inTree) {

    // check if it is labled if it is throw an exception we can not do that
    if(isLabeled) {
      throw std::runtime_error("Can not inject into a labeled tree");
    }

    // inject
    tree->inject(inputIndex, inTree.getPtr());
  }

  // extracts the lambdas from the lambda tree
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal, int32_t &startLabel) {

    // if the lambda tree is not labeled, label it
    if(!isLabeled) {
      tree->labelTree(startLabel);
    }

    // extract the lambdas from this lambda tree and
    extractLambdas(returnVal, tree);
  }

  std::string toTCAPString(int &startLabel,
                           const std::string& computationName,
                           int computationLabel,
                           bool whetherToRemoveUnusedOutputColumns,
                           MultiInputsBase *multiInputsComp,
                           bool shouldFilter = false) {

    // if the lambda tree is not labeled, label it
    if(!isLabeled) {
      tree->labelTree(startLabel);
    }

    // generate the tcap strings for atomic computations
    std::vector<std::string> tcapStrings;
    tree->getTCAPStrings(computationLabel,
                         computationName,
                         true,
                         multiInputsComp,
                         shouldFilter,
                         tcapStrings);

    // clear the input the output is all in a single tuple set now
    multiInputsComp->resize((int) tree->getOutputColumns().size());

    // set the inputs
    multiInputsComp->inputNames = tree->getOutputColumns();
    for(int i = 0; i < tree->getOutputColumns().size(); ++i) {
      multiInputsComp->tupleSetNamesForInputs[i] = tree->getOutputTupleSetName();
      multiInputsComp->inputColumnsToApplyForInputs[i] = tree->getGeneratedColumns();
      multiInputsComp->inputColumnsForInputs[i] = tree->getOutputColumns();
    }

    // combine all the tcap strings
    std::string outputTCAPString;
    for (const auto &tcapString : tcapStrings) {
      outputTCAPString.append(tcapString);
    }

    return outputTCAPString;
  }

};

}

#endif