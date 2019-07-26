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
    std::shared_ptr<DereferenceLambda<ReturnType>>
        newRoot = std::make_shared<DereferenceLambda<ReturnType>>(treeWithPointer);
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
  std::string toTCAPString(std::string inputTupleSetName,
                           std::vector<std::string> &columnNames,
                           std::vector<std::string> &columnsToApply,
                           std::vector<std::string> &childrenLambdas,
                           int &lambdaLabel,
                           const std::string& computationName,
                           int computationLabel,
                           std::string &outputTupleSetName,
                           std::vector<std::string> &outputColumnNames,
                           std::string &addedOutputColumnName,
                           std::string &myLambdaName,
                           bool whetherToRemoveUnusedOutputColumns,
                           MultiInputsBase *multiInputsComp = nullptr,
                           bool amIPartOfJoinPredicate = false) {

    std::vector<std::string> tcapStrings;
    std::vector<std::string> inputTupleSetNames = { std::move(inputTupleSetName) };
    tree->toTCAPString(tcapStrings,
                       inputTupleSetNames,
                       columnNames,
                       columnsToApply,
                       childrenLambdas,
                       lambdaLabel,
                       computationName,
                       computationLabel,
                       addedOutputColumnName,
                       myLambdaName,
                       outputTupleSetName,
                       multiInputsComp,
                       amIPartOfJoinPredicate);

    // TODO need figure out what is happening here
    bool isOutputInInput = false;
    outputColumnNames.clear();
    if (!whetherToRemoveUnusedOutputColumns) {

      for (const auto &columnName : columnNames) {
        outputColumnNames.push_back(columnName);
        if (addedOutputColumnName == columnName) {
          isOutputInInput = true;
        }
      }

      if (!isOutputInInput) {
        outputColumnNames.push_back(addedOutputColumnName);
      }

    } else {
      outputColumnNames.push_back(addedOutputColumnName);
    }

    // TODO this is very dirty and should not be done like that! For now I'm going to patch it!
    if (whetherToRemoveUnusedOutputColumns) {

      // get the last tcap string
      unsigned long last = tcapStrings.size() - 1;

      PDB_COUT << "tcapStrings[" << last << "]=" << tcapStrings[last] << std::endl;
      std::string right = tcapStrings[last].substr(tcapStrings[last].find("<="));

      // by default the end is an empty string
      std::string end;

      // check if we have an info dictionary if we have chop off the end and store it in the end variable
      if (right.find('[') != std::string::npos) {
        end = right.substr(right.find('['));
        right = right.substr(0, right.find('['));
      }

      // find the positions of the last brackets ()
      unsigned long pos1 = right.find_last_of('(');
      unsigned long pos2 = right.rfind("),");

      // empty out anything between the brackets
      right.replace(pos1 + 1, pos2 - 1 - (pos1 + 1) + 1, "");

      // combine the string and replace it
      tcapStrings[last] = outputTupleSetName + " (" + addedOutputColumnName + ") " + right + end;
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