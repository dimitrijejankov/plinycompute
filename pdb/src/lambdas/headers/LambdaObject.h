//
// Created by dimitrije on 2/19/19.
//

#ifndef PDB_GENERICLAMBDAOBJECT_H
#define PDB_GENERICLAMBDAOBJECT_H

#include <memory>
#include <vector>
#include <functional>
#include <mustache.h>
#include <mustache_helper.h>
#include "Object.h"
#include "Handle.h"
#include "Ptr.h"
#include "TupleSpec.h"
#include "ComputeExecutor.h"
#include "ComputeInfo.h"
#include "MultiInputsBase.h"
#include "TupleSetMachine.h"
#include "LambdaTree.h"

namespace pdb {

class LambdaObject;
typedef std::shared_ptr<LambdaObject> LambdaObjectPtr;

// this is the base class from which all pdb :: Lambdas derive
class LambdaObject {

 public:

  virtual ~LambdaObject() = default;

  // this gets an executor that appends the result of running this lambda to the end of each tuple
  virtual ComputeExecutorPtr getExecutor(TupleSpec &inputSchema,
                                         TupleSpec &attsToOperateOn,
                                         TupleSpec &attsToIncludeInOutput) = 0;

  // this gets an executor that appends the result of running this lambda to the end of each tuple; also accepts a parameter
  // in the default case the parameter is ignored and the "regular" version of the executor is created
  virtual ComputeExecutorPtr getExecutor(TupleSpec &inputSchema,
                                         TupleSpec &attsToOperateOn,
                                         TupleSpec &attsToIncludeInOutput,
                                         const ComputeInfoPtr &) {
    return getExecutor(inputSchema, attsToOperateOn, attsToIncludeInOutput);
  }

  // this gets an executor that appends a hash value to the end of each tuple; implemented, for example, by ==
  virtual ComputeExecutorPtr getLeftHasher(TupleSpec &inputSchema,
                                           TupleSpec &attsToOperateOn,
                                           TupleSpec &attsToIncludeInOutput) {
    std::cout << "getLeftHasher not implemented for this type!!\n";
    exit(1);
  }

  // version of the above that accepts ComputeInfo
  virtual ComputeExecutorPtr getLeftHasher(TupleSpec &inputSchema,
                                           TupleSpec &attsToOperateOn,
                                           TupleSpec &attsToIncludeInOutput,
                                           const ComputeInfoPtr &) {
    return getLeftHasher(inputSchema, attsToOperateOn, attsToIncludeInOutput);
  }

  // this gets an executor that appends a hash value to the end of each tuple; implemented, for example, by ==
  virtual ComputeExecutorPtr getRightHasher(TupleSpec &inputSchema,
                                            TupleSpec &attsToOperateOn,
                                            TupleSpec &attsToIncludeInOutput) {
    std::cout << "getRightHasher not implemented for this type!!\n";
    exit(1);
  }

  // version of the above that accepts ComputeInfo
  virtual ComputeExecutorPtr getRightHasher(TupleSpec &inputSchema,
                                            TupleSpec &attsToOperateOn,
                                            TupleSpec &attsToIncludeInOutput,
                                            const ComputeInfoPtr &) {
    return getRightHasher(inputSchema, attsToOperateOn, attsToIncludeInOutput);
  }

  virtual unsigned int getNumInputs() = 0;

  virtual unsigned int getInputIndex(int i) {
    if (i >= this->getNumInputs()) {
      return (unsigned int) (-1);
    }
    return inputIndexes[i];
  }

  /**
  * Used to set the index of this lambda's input in the corresponding computation
  * @param i - the index of the input
  * @param index - the index of the input in the corresponding computation
  */
  void setInputIndex(int i, unsigned int index) {
    size_t numInputs = this->getNumInputs();
    if (numInputs == 0) {
      numInputs = 1;
    }
    if (inputIndexes.size() != numInputs) {
      inputIndexes.resize(numInputs);
    }
    if (i < numInputs) {
      this->inputIndexes[i] = index;
    }
  }

  // returns the name of this LambdaBase type, as a string
  virtual std::string getTypeOfLambda() = 0;

  // one big technical problem is that when tuples are added to a hash table to be recovered
  // at a later time, we we break a pipeline.  The difficulty with this is that when we want
  // to probe a hash table to find a set of hash values, we can't use the input TupleSet as
  // a way to create the columns to store the result of the probe.  The hash table needs to
  // be able to create (from scratch) the columns that store the output.  This is a problem,
  // because the hash table has no information about the types of the objects that it contains.
  // The way around this is that we have a function attached to each LambdaObject that allows
  // us to ask the LambdaObject to try to add a column to a tuple set, of a specific type,
  // where the type name is specified as a string.  When the hash table needs to create an output
  // TupleSet, it can ask all of the GenericLambdaObjects associated with a query to create the
  // necessary columns, as a way to build up the output TupleSet.  This method is how the hash
  // table can ask for this.  It takes tree args: the type  of the column that the hash table wants
  // the tuple set to build, the tuple set to add the column to, and the position where the
  // column will be added.  If the LambdaObject cannot build the column (it has no knowledge
  // of that type) a false is returned.  Otherwise, a true is returned.
  //virtual bool addColumnToTupleSet (std :: string &typeToMatch, TupleSetPtr addToMe, int posToAddTo) = 0;

  // returns the number of children of this Lambda type
  virtual int getNumChildren() = 0;

  // gets a particular child of this Lambda
  virtual LambdaObjectPtr getChild(int which) = 0;

  // returns a string containing the type that is returned when this lambda is executed
  virtual std::string getOutputType() = 0;

  /**
   * takes inputTupleSetName, inputColumnNames, inputColumnsToApply, outputTupleSetName,
   * outputColumnName, outputColumns, TCAP operation name as inputs, and outputs a TCAP string
   * with one TCAP operation.
   *
   * @param inputTupleSetName - // TODO add proper descriptions of the parameter
   * @param inputColumnNames - // TODO add proper descriptions of the parameter
   * @param inputColumnsToApply - // TODO add proper descriptions of the parameter
   * @param outputTupleSetName - // TODO add proper descriptions of the parameters
   * @param outputColumns - // TODO add proper descriptions of the parameters
   * @param outputColumnName - // TODO add proper descriptions of the parameters
   * @param tcapOperation - // TODO add proper descriptions of the parameters
   * @param computationNameAndLabel - // TODO add proper descriptions of the parameters
   * @param lambdaNameAndLabel - // TODO add proper descriptions of the parameters
   * @return the generated tcap string - // TODO add proper descriptions of the parameters
   */
  std::string getTCAPString(const std::string &inputTupleSetName,
                            const std::vector<std::string> &inputColumnNames,
                            const std::vector<std::string> &inputColumnsToApply,
                            const std::string &outputTupleSetName,
                            const std::vector<std::string> &outputColumns,
                            const std::string &outputColumnName,
                            const std::string &tcapOperation,
                            const std::string &computationNameAndLabel,
                            const std::string &lambdaNameAndLabel,
                            const std::map<std::string, std::string> &info) {

    mustache::mustache outputTupleSetNameTemplate
        {"{{outputTupleSetName}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}) <= "
         "{{tcapOperation}} ({{inputTupleSetName}}({{#inputColumnsToApply}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnsToApply}}), "
         "{{inputTupleSetName}}({{#hasColumnNames}}{{#inputColumnNames}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumnNames}}{{/hasColumnNames}}), "
         "'{{computationNameAndLabel}}', "
         "{{#hasLambdaNameAndLabel}}'{{lambdaNameAndLabel}}', {{/hasLambdaNameAndLabel}}"
         "[{{#info}}('{{key}}', '{{value}}'){{^isLast}}, {{/isLast}}{{/info}}])\n"};

    // create the data for the output columns
    mustache::data outputColumnData = mustache::from_vector<std::string>(outputColumns);

    // create the data for the input columns to apply
    mustache::data inputColumnsToApplyData = mustache::from_vector<std::string>(inputColumnsToApply);

    // create the data for the input columns to apply
    mustache::data inputColumnNamesData = mustache::from_vector<std::string>(inputColumnNames);

    // create the info data
    mustache::data infoData = mustache::from_map(info);

    // create the data for the lambda
    mustache::data lambdaData;

    lambdaData.set("outputTupleSetName", outputTupleSetName);
    lambdaData.set("outputColumns", outputColumnData);
    lambdaData.set("tcapOperation", tcapOperation);
    lambdaData.set("inputTupleSetName", inputTupleSetName);
    lambdaData.set("inputColumnsToApply", inputColumnsToApplyData);
    lambdaData.set("hasColumnNames", !inputColumnNames.empty());
    lambdaData.set("inputColumnNames", inputColumnNamesData);
    lambdaData.set("inputTupleSetName", inputTupleSetName);
    lambdaData.set("computationNameAndLabel", computationNameAndLabel);
    lambdaData.set("hasLambdaNameAndLabel", !lambdaNameAndLabel.empty());
    lambdaData.set("lambdaNameAndLabel", lambdaNameAndLabel);
    lambdaData.set("info", infoData);

    return outputTupleSetNameTemplate.render(lambdaData);
  }

  virtual std::string toTCAPStringForCartesianJoin(int lambdaLabel,
                                                   std::string computationName,
                                                   int computationLabel,
                                                   std::string &outputTupleSetName,
                                                   std::vector<std::string> &outputColumns,
                                                   std::string &outputColumnName,
                                                   std::string &myLambdaName,
                                                   MultiInputsBase *multiInputsComp) {
    std::cout << "toTCAPStringForCartesianJoin() should not be implemented here!" << std::endl;
    exit(1);
  }

  virtual std::string toTCAPString(std::vector<std::string> &inputTupleSetNames,
                                   std::vector<std::string> &inputColumnNames,
                                   std::vector<std::string> &inputColumnsToApply,
                                   std::vector<std::string> &childrenLambdaNames,
                                   int lambdaLabel,
                                   std::string computationName,
                                   int computationLabel,
                                   std::string &outputTupleSetName,
                                   std::vector<std::string> &outputColumns,
                                   std::string &outputColumnName,
                                   std::string &myLambdaName,
                                   MultiInputsBase *multiInputsComp = nullptr,
                                   bool amIPartOfJoinPredicate = false,
                                   bool amILeftChildOfEqualLambda = false,
                                   bool amIRightChildOfEqualLambda = false,
                                   std::string parentLambdaName = "",
                                   bool isSelfJoin = false) {
    std::string tcapString;
    std::string lambdaType = getTypeOfLambda();
    if ((lambdaType.find("==") != std::string::npos) ||
        (lambdaType.find("&&") != std::string::npos)) {
      return "";
    }

    if ((lambdaType.find("native_lambda") != std::string::npos) && (multiInputsComp != nullptr)
        && amIPartOfJoinPredicate &&
        !amIRightChildOfEqualLambda
        && ((parentLambdaName.empty()) || (parentLambdaName.find("&&") != std::string::npos))) {

      return toTCAPStringForCartesianJoin(lambdaLabel,
                                          computationName,
                                          computationLabel,
                                          outputTupleSetName,
                                          outputColumns,
                                          outputColumnName,
                                          myLambdaName,
                                          multiInputsComp);
    }

    std::string computationNameWithLabel = computationName + "_" + std::to_string(computationLabel);
    myLambdaName = getTypeOfLambda() + "_" + std::to_string(lambdaLabel);
    std::string inputTupleSetName = inputTupleSetNames[0];
    std::string tupleSetMidTag = "OutFor";

    std::vector<std::string> originalInputColumnsToApply;

    int myIndex;
    if (multiInputsComp != nullptr) {
      if (amILeftChildOfEqualLambda || amIRightChildOfEqualLambda) {
        tupleSetMidTag = "Extracted";
      }
      myIndex = this->getInputIndex(0);
      PDB_COUT << myLambdaName + ": myIndex=" << myIndex << std::endl;
      inputTupleSetName = multiInputsComp->getTupleSetNameForIthInput(myIndex);
      PDB_COUT << "inputTupleSetName=" << inputTupleSetName << std::endl;
      inputColumnNames = multiInputsComp->getInputColumnsForIthInput(myIndex);

      inputColumnsToApply.clear();

      if (this->getNumInputs() == 1) {
        inputColumnsToApply.push_back(multiInputsComp->getNameForIthInput(myIndex));
        originalInputColumnsToApply.push_back(multiInputsComp->getNameForIthInput(myIndex));
      } else {
        for (int i = 0; i < this->getNumInputs(); i++) {
          int index = this->getInputIndex(i);
          inputColumnsToApply.push_back(multiInputsComp->getNameForIthInput(index));
          originalInputColumnsToApply.push_back(
              multiInputsComp->getNameForIthInput(myIndex));
        }
      }
      multiInputsComp->setLambdasForIthInputAndPredicate(
          myIndex, parentLambdaName, myLambdaName);
    }

    PDB_COUT << "input columns to apply: " << std::endl;
    for (const auto &i : originalInputColumnsToApply) {
      PDB_COUT << i << std::endl;
    }

    outputTupleSetName = lambdaType.substr(0, 5) + "_" + std::to_string(lambdaLabel) + tupleSetMidTag + computationName
        + std::to_string(computationLabel);
    outputColumnName =
        lambdaType.substr(0, 5) + "_" + std::to_string(lambdaLabel) + "_" + std::to_string(computationLabel)
            + tupleSetMidTag;

    outputColumns.clear();
    for (const auto &inputColumnName : inputColumnNames) {
      outputColumns.push_back(inputColumnName);
    }
    outputColumns.push_back(outputColumnName);

    // the additional info about this attribute access lambda
    std::map<std::string, std::string> info;

    // fill in the info
    info["lambdaType"] = getTypeOfLambda();

    tcapString += getTCAPString(inputTupleSetName,
                                inputColumnNames,
                                inputColumnsToApply,
                                outputTupleSetName,
                                outputColumns,
                                outputColumnName,
                                "APPLY",
                                computationNameWithLabel,
                                myLambdaName,
                                info);

    if (multiInputsComp != nullptr) {
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

        std::string hashOperator;
        if (amILeftChildOfEqualLambda) {
          hashOperator = "HASHLEFT";
        } else {
          hashOperator = "HASHRIGHT";
        }
        outputTupleSetName = outputTupleSetName + "_hashed";
        outputColumnName = outputColumnName + "_hash";
        outputColumns.clear();

        for (const auto &inputColumnName : inputColumnNames) {
          outputColumns.push_back(inputColumnName);
        }
        outputColumns.push_back(outputColumnName);

        tcapString += getTCAPString(inputTupleSetName,
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
          PDB_COUT << "curInput is " << curInput << std::endl;
          auto iter = std::find(outputColumns.begin(), outputColumns.end(), curInput);
          if (iter != outputColumns.end()) {
            PDB_COUT << "MultiInputsBase with index=" << index << " is updated."
                     << std::endl;
            multiInputsComp->setTupleSetNameForIthInput(index, outputTupleSetName);
            multiInputsComp->setInputColumnsForIthInput(index, outputColumns);
            multiInputsComp->setInputColumnsToApplyForIthInput(index, outputColumnName);
          }
          PDB_COUT << std::endl;
          auto iter1 = std::find(originalInputColumnsToApply.begin(),
                                 originalInputColumnsToApply.end(),
                                 curInput);
          if (iter1 != originalInputColumnsToApply.end()) {
            PDB_COUT << "MultiInputsBase with index=" << index << " is updated."
                     << std::endl;
            multiInputsComp->setTupleSetNameForIthInput(index, outputTupleSetName);
            multiInputsComp->setInputColumnsForIthInput(index, outputColumns);
            multiInputsComp->setInputColumnsToApplyForIthInput(index, outputColumnName);
          }
        }
      } else {
        // only update myIndex
        multiInputsComp->setTupleSetNameForIthInput(myIndex, outputTupleSetName);
        multiInputsComp->setInputColumnsForIthInput(myIndex, outputColumns);
        multiInputsComp->setInputColumnsToApplyForIthInput(myIndex, outputColumnName);
      }
    }
    return tcapString;
  }

  void toTCAPString(std::vector<std::string> &tcapStrings,
                    std::vector<std::string> &inputTupleSetNames,
                    std::vector<std::string> &inputColumnNames,
                    std::vector<std::string> &inputColumnsToApply,
                    std::vector<std::string> &childrenLambdaNames,
                    int &lambdaLabel,
                    const std::string& computationName,
                    int computationLabel,
                    std::string &addedOutputColumnName,
                    std::string &myLambdaName,
                    std::string &outputTupleSetName,
                    MultiInputsBase *multiInputsComp = nullptr,
                    bool amIPartOfJoinPredicate = false,
                    bool amILeftChildOfEqualLambda = false,
                    bool amIRightChildOfEqualLambda = false,
                    std::string parentLambdaName = "",
                    bool isSelfJoin = false) {

    std::vector<std::string> columnsToApply;
    std::vector<std::string> childrenLambdas;
    std::vector<std::string> inputNames;
    std::vector<std::string> inputColumns;

    if (this->getNumChildren() > 0) {

      // move the input columns to apply
      columnsToApply.swap(inputColumnsToApply);

      // move the children lambdas
      childrenLambdas.swap(childrenLambdaNames);

      for (const auto &inputTupleSetName : inputTupleSetNames) {
        auto iter = std::find(inputNames.begin(), inputNames.end(), inputTupleSetName);
        if (iter == inputNames.end()) {
          inputNames.push_back(inputTupleSetName);
        }
        else {
          std::cout << "This is interesting" << std::endl;
        }
      }

      // move the input tuple set names
      inputTupleSetNames.clear();

      // move the input columns
      inputColumns.swap(inputColumnNames);
    }

    std::string myTypeName = this->getTypeOfLambda();
    PDB_COUT << "\tExtracted lambda named: " << myTypeName << "\n";
    std::string myName = myTypeName + "_" + std::to_string(lambdaLabel + this->getNumChildren());

    bool isLeftChildOfEqualLambda = false;
    bool isRightChildOfEqualLambda = false;
    bool isChildSelfJoin = false;

    LambdaObjectPtr nextChild = nullptr;
    for (int i = 0; i < this->getNumChildren(); i++) {
      LambdaObjectPtr child = this->getChild(i);

      if ((i + 1) < this->getNumChildren()) {
        nextChild = this->getChild(i + 1);
      }

      if (myTypeName == "==") {

        if (i == 0) {
          isLeftChildOfEqualLambda = true;
        }

        if (i == 1) {
          isRightChildOfEqualLambda = true;
        }

      }

      if ((isLeftChildOfEqualLambda || isRightChildOfEqualLambda) && (multiInputsComp != nullptr)) {

        std::string nextInputName;

        if (nextChild != nullptr) {
          nextInputName = multiInputsComp->getNameForIthInput(nextChild->getInputIndex(0));
        }

        std::string myInputName = multiInputsComp->getNameForIthInput(child->getInputIndex(0));
        if (nextInputName == myInputName) {
          isChildSelfJoin = true;
        }
      }

      child->toTCAPString(tcapStrings,
                          inputNames,
                          inputColumns,
                          columnsToApply,
                          childrenLambdas,
                          lambdaLabel,
                          computationName,
                          computationLabel,
                          addedOutputColumnName,
                          myLambdaName,
                          outputTupleSetName,
                          multiInputsComp,
                          amIPartOfJoinPredicate,
                          isLeftChildOfEqualLambda,
                          isRightChildOfEqualLambda,
                          myName,
                          isChildSelfJoin);

      inputColumnsToApply.push_back(addedOutputColumnName);
      childrenLambdaNames.push_back(myLambdaName);

      if (multiInputsComp != nullptr) {
        auto iter = std::find(inputTupleSetNames.begin(), inputTupleSetNames.end(), outputTupleSetName);

        if (iter == inputTupleSetNames.end()) {
          inputTupleSetNames.push_back(outputTupleSetName);
        }

      } else {

        inputTupleSetNames.clear();
        inputTupleSetNames.push_back(outputTupleSetName);
        inputColumnNames.clear();
      }

      for (const auto &inputColumn : inputColumns) {
        auto iter = std::find(inputColumnNames.begin(), inputColumnNames.end(), inputColumn);
        if (iter == inputColumnNames.end()) {
          inputColumnNames.push_back(inputColumn);
        }
      }

      isLeftChildOfEqualLambda = false;
      isRightChildOfEqualLambda = false;
      isChildSelfJoin = false;
      nextChild = nullptr;
    }

    std::vector<std::string> outputColumns;
    std::string tcapString = this->toTCAPString(inputTupleSetNames,
                                                inputColumnNames,
                                                inputColumnsToApply,
                                                childrenLambdaNames,
                                                lambdaLabel,
                                                computationName,
                                                computationLabel,
                                                outputTupleSetName,
                                                outputColumns,
                                                addedOutputColumnName,
                                                myLambdaName,
                                                multiInputsComp,
                                                amIPartOfJoinPredicate,
                                                amILeftChildOfEqualLambda,
                                                amIRightChildOfEqualLambda,
                                                parentLambdaName,
                                                isSelfJoin);

    tcapStrings.push_back(tcapString);
    lambdaLabel++;

    if (multiInputsComp == nullptr) {
      inputTupleSetNames.clear();
      inputTupleSetNames.push_back(outputTupleSetName);
    }

    inputColumnNames.clear();
    for (const auto &outputColumn : outputColumns) {
      inputColumnNames.push_back(outputColumn);
    }
  }

  virtual std::map<std::string, std::string> getInfo() = 0;

 private:
  /**
   * input index in a multi-input computation
   */
  std::vector<unsigned int> inputIndexes;

};
}

#endif //PDB_GENERICLAMBDAOBJECT_H