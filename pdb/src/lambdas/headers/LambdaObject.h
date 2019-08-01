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
#include "LambdaFormatFunctions.h"

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

  // returns all the inputs in the lambda tree
  void getAllInputs(std::set<int32_t> &inputs) {

    // go through all the children
    for (int i = 0; i < this->getNumChildren(); i++) {

      // get the inputs from the children
      this->getChild(i)->getAllInputs(inputs);
    }

    // if we have no children that means we are getting the inputs
    if(this->getNumChildren() == 0) {

      // copy the input indices
      for(const auto &in : inputIndexes) {
        inputs.insert(in);
      }
    }
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

  const std::string &getOutputTupleSetName() const {
    return outputTupleSetName;
  }

  const std::vector<std::string> &getOutputColumns() const {
    return outputColumns;
  }

  const std::vector<std::string> &getAppliedColumns() const {
    return appliedColumns;
  }

  const std::vector<std::string> &getGeneratedColumns() const {
    return generatedColumns;
  }

  static int32_t getInputIndexForColumns(MultiInputsBase *multiInputsComp, const std::vector<std::string> &requiredColumns) {

    // grab the inputs
    auto &inputs = multiInputsComp->inputColumnsForInputs;

    // try to find the columns in one of the inputs
    auto it = std::find_if(inputs.begin(), inputs.end(), [&](auto &columns) {

      // check if all the columns are here
      return std::all_of(requiredColumns.begin(), requiredColumns.end(), [&](auto &column) {
        return std::find(columns.begin(), columns.end(), column) != columns.end();
      });
    });

    // this must always find something or something has gone wrong
    if(it == inputs.end()) {
      return -1;
    }

    // set the index
    return std::distance(inputs.begin(), it);
  }

  static bool isJoined(const std::set<int32_t> &a, const std::set<int32_t> &b) {
    return std::any_of(a.begin(), a.end(), [&](const auto inputIndex) {
      return b.find(inputIndex) != b.end();
    });
  }

  virtual std::string toTCAPString(std::vector<std::string> &childrenLambdaNames,
                                   int lambdaLabel,
                                   const std::string &computationName,
                                   int computationLabel,
                                   std::string &myLambdaName,
                                   MultiInputsBase *multiInputsComp,
                                   bool shouldFilter,
                                   const std::string &parentLambdaName) {

    // create the data for the lambda so we can generate the strings
    mustache::data lambdaData;
    lambdaData.set("computationName", computationName);
    lambdaData.set("computationLabel", std::to_string(computationLabel));
    lambdaData.set("typeOfLambda", getTypeOfLambda());
    lambdaData.set("lambdaLabel", std::to_string(lambdaLabel));
    lambdaData.set("tupleSetMidTag", "OutFor");

    // create the computation name with label
    mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
    std::string computationNameWithLabel = computationNameWithLabelTemplate.render(lambdaData);

    // create the lambda name
    mustache::mustache lambdaNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}"};
    myLambdaName = lambdaNameTemplate.render(lambdaData);
    multiInputsComp->setLambdasForIthInputAndPredicate(this->getInputIndex(0), parentLambdaName, myLambdaName);

    // create the output tuple set name
    mustache::mustache outputTupleSetNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}{{tupleSetMidTag}}{{computationName}}{{computationLabel}}"};
    outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

    // create the output columns
    mustache::mustache outputColumnNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}_{{computationLabel}}_{{tupleSetMidTag}}"};
    generatedColumns = { outputColumnNameTemplate.render(lambdaData) };

    // the input tuple set
    std::string inputTupleSet;

    // we are preparing the input columns and the columns we want to apply
    std::vector<std::string> inputColumnNames;
    appliedColumns.clear();

    // if this lambda has no children that means it gets the input columns directly from the input tuple set,
    // otherwise it is getting it from the child lambda.
    int currIndex;
    if(getNumChildren() != 0) {

      // this lambda generation is designed for only one input
      assert(this->getNumChildren() == 1);

      // get the child lambda
      LambdaObjectPtr child = this->getChild(0);

      // grab what columns we require
      appliedColumns = child->generatedColumns;

      // get the index
      currIndex = getInputIndexForColumns(multiInputsComp, appliedColumns);
      assert(currIndex != -1);
    }
    else {

      // make sure that all of the inputs are in the same tuple set
      currIndex = this->getInputIndex(0);
      bool isJoined = std::all_of(inputIndexes.begin(), inputIndexes.end(), [&](auto in) {
        return multiInputsComp->joinGroupForInput[currIndex] == multiInputsComp->joinGroupForInput[in];
      });

      // this has to be true you can not have multiple tuple sets as an input to this lambda
      assert(isJoined);

      // go through all the inputs
      for(int i = 0; i < getNumInputs(); ++i) {

        // insert the columns
        appliedColumns.emplace_back(multiInputsComp->inputNames[this->getInputIndex(i)]);
      }
    }

    // get the name of the input tuple set
    inputTupleSet = multiInputsComp->tupleSetNamesForInputs[currIndex];

    // the inputs that that are forwarded
    auto &inputs = multiInputsComp->inputColumnsForInputs[currIndex];
    std::for_each(inputs.begin(), inputs.end(), [&](const auto &column) {

      // check if we are supposed to keep this input column, we keep it either if we are not a root, since it may be used later
      // or if it is a root and it is requested to be kept at the output
      if(!isRoot || multiInputsComp->inputColumnsToKeep.find(column) != multiInputsComp->inputColumnsToKeep.end()) {

        // if we are supposed to keep this column add it to the input columns
        inputColumnNames.emplace_back(column);
      }
    });

    // the output columns are basically the input columns we are keeping from the input tuple set, with the output column of this lambda
    outputColumns = inputColumnNames;
    outputColumns.emplace_back(generatedColumns[0]);

    // generate the TCAP string for the lambda
    std::string tcapString;
    tcapString += formatLambdaComputation(inputTupleSet,
                                          inputColumnNames,
                                          appliedColumns,
                                          outputTupleSetName,
                                          outputColumns,
                                          "APPLY",
                                          computationNameWithLabel,
                                          myLambdaName,
                                          getInfo());

    // if we are a part of the join predicate
    if(shouldFilter) {

      // mark as filtered
      isFiltered = true;

      // the previous lambda that created the boolean column is the input to the filter
      std::string inputTupleSetName = outputTupleSetName;

      // this basically removes "_BOOL" column from the output columns since we are done with it after the filter
      outputColumns.pop_back();

      // make the name for the new tuple set that is the output of the filter
      outputTupleSetNameTemplate = {"FILTERED_{{lambdaLabel}}{{tupleSetMidTag}}{{computationName}}{{computationLabel}}"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // we are applying the filtering on the boolean column
      appliedColumns = { generatedColumns[0] };

      // we are not generating any columns after the filter
      generatedColumns = {};

      // make the filter
      tcapString += formatFilterComputation(outputTupleSetName,
                                            outputColumns,
                                            inputTupleSetName,
                                            appliedColumns,
                                            inputColumnNames,
                                            computationNameWithLabel);
    }

    // update the join group
    joinGroup = multiInputsComp->joinGroupForInput[currIndex];

    // go through each tuple set and update stuff
    for(int i = 0; i < multiInputsComp->tupleSetNamesForInputs.size(); ++i) {

      // check if this tuple set is the same index
      if(multiInputsComp->joinGroupForInput[i] == joinGroup) {

        // the output tuple set is the new set with these columns
        multiInputsComp->tupleSetNamesForInputs[i] = outputTupleSetName;
        multiInputsComp->inputColumnsForInputs[i] = outputColumns;
        multiInputsComp->inputColumnsToApplyForInputs[i] = generatedColumns;

        // this input was joined
        joinedInputs.insert(i);
      }
    }

    // return the tcap string
    return std::move(tcapString);
  }

  void getJoinedInputs(std::vector<std::string> &tcapStrings,
                       const std::set<int32_t> &inputs,
                       MultiInputsBase *multiInputsComp) {


  }

  void getTCAPStrings(std::vector<std::string> &tcapStrings,
                      int &lambdaLabel,
                      const std::string &computationName,
                      int computationLabel,
                      std::string &myLambdaName,
                      MultiInputsBase *multiInputsComp = nullptr,
                      bool shouldFilter = false,
                      const std::string &parentLambdaName = "") {

    //  and empty parent lambda name means that this is the root of a lambda tree
    isRoot = parentLambdaName.empty();

    // make the name for this lambda object
    std::string myTypeName = this->getTypeOfLambda();
    std::string myName = myTypeName + "_" + std::to_string(lambdaLabel + this->getNumChildren());

    // should the child filter or not
    bool shouldFilterChild = false;
    if(myTypeName == "and" || myTypeName == "deref") {
      shouldFilterChild = shouldFilter;
    }

    // the names of the child lambda we need that for the equal lambda etc..
    std::vector<std::string> childrenLambdas;

    LambdaObjectPtr nextChild = nullptr;
    for (int i = 0; i < this->getNumChildren(); i++) {

      // get the child lambda
      LambdaObjectPtr child = this->getChild(i);

      // get the next child if it exists
      if ((i + 1) < this->getNumChildren()) {
        nextChild = this->getChild(i + 1);
      }

      // recurse to generate the TCAP string
      child->getTCAPStrings(tcapStrings,
                            lambdaLabel,
                            computationName,
                            computationLabel,
                            myLambdaName,
                            multiInputsComp,
                            shouldFilterChild,
                            myName);

      childrenLambdas.push_back(myLambdaName);
      nextChild = nullptr;
    }

    // generate the TCAP string for the current lambda
    std::string tcapString = this->toTCAPString(childrenLambdas,
                                                lambdaLabel,
                                                computationName,
                                                computationLabel,
                                                myLambdaName,
                                                multiInputsComp,
                                                shouldFilter,
                                                parentLambdaName);

    tcapStrings.push_back(tcapString);
    lambdaLabel++;
  }

  virtual std::map<std::string, std::string> getInfo() = 0;

  /**
   * input index in a multi-input computation
   */
  std::vector<unsigned int> inputIndexes;

   /**
    * Is the lambda object the root of the lambda tree
    */
   bool isRoot = false;

   /**
    * Has this lambda been followed by a filter
    */
   bool isFiltered = false;

   /**
    * This is telling us what inputs were joined at the time this lambda was processed
    */
   std::set<int32_t> joinedInputs;

   /**
    * The join group the inputs belong to
    */
   int32_t joinGroup = -1;

   /**
    * The name of the generated tuple set
    */
   std::string outputTupleSetName;

   /**
    * The output columns of this lambda
    */
   std::vector<std::string> outputColumns;

   /**
    * The columns this lambda has applied to generate the output
    */
   std::vector<std::string> appliedColumns;

   /**
    * The columns this lambda has generated
    */
   std::vector<std::string> generatedColumns;
};
}

#endif //PDB_GENERICLAMBDAOBJECT_H