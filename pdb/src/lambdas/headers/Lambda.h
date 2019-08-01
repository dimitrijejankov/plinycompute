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

 private:

  // in case we wrap up a non-pointer type
  std::shared_ptr<TypedLambdaObject<ReturnType>> tree;

  // does the actual tree traversal
  // JiaNote: I changed below method from pre-order traversing to post-order traversing, so that it follows the lambda execution ordering
  static void traverse(std::map<std::string, LambdaObjectPtr> &fillMe,
                       LambdaObjectPtr root, int &startLabel) {

    for (int i = 0; i < root->getNumChildren(); i++) {
      LambdaObjectPtr child = root->getChild(i);
      traverse(fillMe, child, startLabel);
    }
    std::string myName = root->getTypeOfLambda();
    myName = myName + "_" + std::to_string(startLabel);
    std::cout << "\tExtracted lambda named: " << myName << "\n";
    startLabel++;
    fillMe[myName] = root;
  }

  /**
 *
 * @param allInputs
 * @param root
 * @param multiInputsBase
 */
  void getInputs(std::vector<std::string> &allInputs, LambdaObjectPtr root, MultiInputsBase *multiInputsBase) {

    for (int i = 0; i < root->getNumChildren(); i++) {

      LambdaObjectPtr child = root->getChild(i);
      getInputs(allInputs, child, multiInputsBase);
    }

    if (root->getNumChildren() == 0) {
      for (int i = 0; i < root->getNumInputs(); i++) {
        std::string myName = multiInputsBase->getNameForIthInput(root->getInputIndex(i));
        auto iter = std::find(allInputs.begin(), allInputs.end(), myName);

        if (iter == allInputs.end()) {
          allInputs.push_back(myName);
        }
      }
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

  std::vector<std::string> getAllInputs(MultiInputsBase *multiInputsBase) {
    std::vector<std::string> ret;
    this->getInputs(ret, tree, multiInputsBase);
    return ret;
  }

  // convert one of these guys to a map
  void toMap(std::map<std::string, LambdaObjectPtr> &returnVal, int &suffix) {
    traverse(returnVal, tree, suffix);
  }

  //This is to get the TCAPString for this lambda tree
  std::string toTCAPString(std::vector<std::string> &childrenLambdas,
                           int &lambdaLabel,
                           const std::string& computationName,
                           int computationLabel,
                           std::string &myLambdaName,
                           bool whetherToRemoveUnusedOutputColumns,
                           MultiInputsBase *multiInputsComp,
                           bool shouldFilter = false) {

    // generate the tcap strings for atomic computations
    std::vector<std::string> tcapStrings;
    tree->getTCAPStrings(tcapStrings,
                         lambdaLabel,
                         computationName,
                         computationLabel,
                         myLambdaName,
                         multiInputsComp,
                         shouldFilter);

    // clear the input the output is all in a single tuple set now
    multiInputsComp->resize(1);

    // set the inputs
    multiInputsComp->tupleSetNamesForInputs[0] = tree->getOutputTupleSetName();
    multiInputsComp->inputNames = tree->getOutputColumns();
    multiInputsComp->inputColumnsToApplyForInputs[0] = tree->getGeneratedColumns();
    multiInputsComp->inputColumnsForInputs[0] = tree->getOutputColumns();

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