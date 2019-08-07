
#pragma once

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

  /**
   * This method labels all the lambdas in the lambda tree with an unique identifier
   * @param label - the label we start labeling the tree
   */
  virtual void labelTree(int32_t &label) {

    // set my label
    myLambdaLabel = label++;

    // set the label of the children
    for (const auto &child : children) {
      child.second->labelTree(label);
    }
  }

  virtual unsigned int getInputIndex(int i) {
    if (i >= this->getNumInputs()) {
      return (unsigned int) (-1);
    }
    return inputIndexes[i];
  }

  /**
   * Returns all the inputs in the lambda tree
   * @param inputs - all the inputs
   */
  void getAllInputs(std::set<int32_t> &inputs) {

    // go through all the children
    for (const auto &child : children) {

      // get the inputs from the children
      child.second->getAllInputs(inputs);
    }

    // if we have no children that means we are getting the inputs
    if (this->children.empty()) {

      // copy the input indices
      for (const auto &in : inputIndexes) {
        inputs.insert(in.second);
      }
    }
  }

  // Used to set the index of this lambda's input in the corresponding computation
  void setInputIndex(int i, unsigned int index) {
    size_t numInputs = this->getNumInputs();
    if (numInputs == 0) {
      numInputs = 1;
    }

    this->inputIndexes[i] = index;
  }

  // returns the name of this LambdaBase type, as a string
  virtual std::string getTypeOfLambda() const = 0;

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

  void inject(int32_t inputIndex, const LambdaObjectPtr &lambdaToInject) {

    // set the label of the children
    for (const auto &child : children) {
      child.second->inject(inputIndex, lambdaToInject);
    }

    // mark this lambda tree
    auto idx = std::find_if (inputIndexes.begin(), inputIndexes.end(), [&](auto idx) { return idx.second == inputIndex; });
    if(idx != inputIndexes.end()) {
      children[idx->first] = lambdaToInject;
      inputIndexes.erase(idx);
    }
  }

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

  std::string getLambdaName() const {

    // create the data for the lambda so we can generate the strings
    mustache::data lambdaData;
    lambdaData.set("typeOfLambda", getTypeOfLambda());
    lambdaData.set("lambdaLabel", std::to_string(myLambdaLabel));

    // return the lambda name
    mustache::mustache lambdaNameTemplate{"{{typeOfLambda}}_{{lambdaLabel}}"};
    return std::move(lambdaNameTemplate.render(lambdaData));
  }

  /**
   * This method is used to generate TCAP for the :
   * - AttAccessLambda
   * - CPlusPlusLambda (it is called from it, there is extra code to join stuff if needed)
   * - DereferenceLambda
   * - MethodCallLambda
   * - SelfLambda
   *
   * It basically creates an apply atomic computation for the lambda. In order for it to do it it needs to have all
   * the inputs in the same tuple set, meaning it can not work on two tuple sets at once.
   *
   * @param multiInputsComp - all the inputs sets that are currently there
   * @param isPredicate - is this a predicate and we need to generate a filter?
   * @return - the TCAP string
   */
  virtual std::string generateTCAPString(MultiInputsBase *multiInputsComp, bool isPredicate) {

    // create the data for the lambda so we can generate the strings
    mustache::data lambdaData;
    lambdaData.set("computationName", myComputationName);
    lambdaData.set("computationLabel", std::to_string(myComputationLabel));
    lambdaData.set("typeOfLambda", getTypeOfLambda());
    lambdaData.set("lambdaLabel", std::to_string(myLambdaLabel));
    lambdaData.set("tupleSetMidTag", myPrefix);

    // create the computation name with label
    mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
    std::string computationNameWithLabel = computationNameWithLabelTemplate.render(lambdaData);

    // create the output tuple set name
    mustache::mustache outputTupleSetNameTemplate
        {"{{tupleSetMidTag}}_{{typeOfLambda}}_{{lambdaLabel}}{{computationName}}{{computationLabel}}"};
    outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

    // create the output columns
    mustache::mustache outputColumnNameTemplate{"{{tupleSetMidTag}}_{{typeOfLambda}}_{{lambdaLabel}}_{{computationLabel}}"};
    generatedColumns = {outputColumnNameTemplate.render(lambdaData)};

    // the input tuple set
    std::string inputTupleSet;

    // we are preparing the input columns and the columns we want to apply
    std::vector<std::string> inputColumnNames;
    appliedColumns.clear();

    // go through the inputs
    int currIndex = -1;
    for(int i = 0; i < getNumInputs(); ++i) {

      // check if we have a child
      auto it = children.find(i);
      if(it != children.end()) {

        // get the child lambda
        LambdaObjectPtr child = it->second;

        // grab what columns we require
        appliedColumns = child->generatedColumns;

        // get the index
        assert(!child->joinedInputs.empty());
        assert(i == 0 || multiInputsComp->joinGroupForInput[currIndex] == multiInputsComp->joinGroupForInput[*child->joinedInputs.begin()]);

        // set the current index
        currIndex = *child->joinedInputs.begin();

        // go to the next input
        continue;
      }

      // check if we have an input if we don't have a child
      auto jt = inputIndexes.find(i);
      if(jt != inputIndexes.end()) {

        assert(i == 0 || multiInputsComp->joinGroupForInput[currIndex] == multiInputsComp->joinGroupForInput[jt->second]);

        // set the current index
        currIndex = jt->second;

        // insert the columns
        appliedColumns.emplace_back(multiInputsComp->inputNames[jt->second]);

        // continue to the next input
        continue;
      }

      // throw an error
      throw std::runtime_error("Don't have an input for index " + std::to_string(i) + ".");
    }

    // get the name of the input tuple set
    inputTupleSet = multiInputsComp->tupleSetNamesForInputs[currIndex];

    // the inputs that that are forwarded
    auto &inputs = multiInputsComp->inputColumnsForInputs[currIndex];
    std::for_each(inputs.begin(), inputs.end(), [&](const auto &column) {

      // check if we are supposed to keep this input column, we keep it either if we are not a root, since it may be used later
      // or if it is a root and it is requested to be kept at the output
      if (!isRoot || multiInputsComp->inputColumnsToKeep.find(column) != multiInputsComp->inputColumnsToKeep.end()) {

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
                                          getLambdaName(),
                                          getInfo());

    // if we are a part of the join predicate
    if (isPredicate) {

      // mark as filtered
      isFiltered = true;

      // the previous lambda that created the boolean column is the input to the filter
      std::string inputTupleSetName = outputTupleSetName;

      // this basically removes "_BOOL" column from the output columns since we are done with it after the filter
      outputColumns.pop_back();

      // make the name for the new tuple set that is the output of the filter
      outputTupleSetNameTemplate =
          {"{{tupleSetMidTag}}_FILTERED_{{lambdaLabel}}{{computationName}}{{computationLabel}}"};
      outputTupleSetName = outputTupleSetNameTemplate.render(lambdaData);

      // we are applying the filtering on the boolean column
      appliedColumns = {generatedColumns[0]};

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
    for (int i = 0; i < multiInputsComp->tupleSetNamesForInputs.size(); ++i) {

      // check if this tuple set is the same index
      if (multiInputsComp->joinGroupForInput[i] == joinGroup) {

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

  /**
   * This creates a bunch of cartesian joins to make sure all inputs are in the same tuple set.
   *
   * @param tcapStrings - the list of generated tcap strings
   * @param inputs - the inputs we have to make sure are joined
   * @param multiInputsComp - the current input tuple sets
   */
  void generateJoinedInputs(std::vector<std::string> &tcapStrings,
                            const std::set<int32_t> &inputs,
                            MultiInputsBase *multiInputsComp) {

    // make sure we actually have inputs
    if (inputs.empty()) {
      return;
    }

    // grab the index of the first input
    int32_t firstInput = *inputs.begin();

    // go through all the inputs
    int32_t idx = 0;
    for (int in : inputs) {

      // skip if joined
      if (multiInputsComp->joinGroupForInput[in] == multiInputsComp->joinGroupForInput[firstInput]) {
        continue;
      }

      // create the data for the lambda
      mustache::data lambdaData;
      lambdaData.set("computationName", myComputationName);
      lambdaData.set("computationLabel", std::to_string(myComputationLabel));
      lambdaData.set("typeOfLambda", getTypeOfLambda());
      lambdaData.set("lambdaLabel", std::to_string(myLambdaLabel));
      lambdaData.set("idx", std::to_string(idx));
      lambdaData.set("tupleSetMidTag", myPrefix);

      // create the computation name with label
      mustache::mustache computationNameWithLabelTemplate{"{{computationName}}_{{computationLabel}}"};
      std::string computationNameWithLabel = computationNameWithLabelTemplate.render(lambdaData);

      /**
       * 1.1 Create a hash one for the LHS side
       */

      // get the index of the left input, any will do since all joined tuple sets are the same
      auto lhsIndex = firstInput;
      auto lhsColumnNames = multiInputsComp->inputColumnsForInputs[lhsIndex];

      // added the lhs attribute
      lambdaData.set("LHSApplyAttribute", lhsColumnNames[0]);

      // we need a cartesian join hash-one for lhs
      const std::string &leftTupleSetName = multiInputsComp->tupleSetNamesForInputs[lhsIndex];

      // the lhs column can be any column we only need it to get the number of rows
      std::vector<std::string> leftColumnsToApply = {lhsColumnNames[0]};

      // make the output tuple set
      mustache::mustache leftOutputTupleTemplate
          {"{{tupleSetMidTag}}_hashOneFor_{{LHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}_{{idx}}"};
      std::string leftOutputTupleSetName = leftOutputTupleTemplate.render(lambdaData);

      // make the column name
      mustache::mustache leftOutputColumnNameTemplate{"OneFor_left_{{computationLabel}}_{{lambdaLabel}}_{{idx}}"};
      std::string leftOutputColumnName = leftOutputColumnNameTemplate.render(lambdaData);

      // make the columns
      std::vector<std::string> leftOutputColumns = lhsColumnNames;
      leftOutputColumns.push_back(leftOutputColumnName);

      // make the lambda
      std::string tcapString = formatLambdaComputation(leftTupleSetName,
                                                       lhsColumnNames,
                                                       leftColumnsToApply,
                                                       leftOutputTupleSetName,
                                                       leftOutputColumns,
                                                       "HASHONE",
                                                       computationNameWithLabel,
                                                       "",
                                                       {});

      /**
       * 1.2 Create a hash one for the RHS side
       */

      // get the index of the right input, any will do since all joined tuple sets are the same
      auto rhsIndex = in;
      auto rhsColumnNames = multiInputsComp->inputColumnsForInputs[rhsIndex];

      lambdaData.set("RHSApplyAttribute", rhsColumnNames[0]);

      // we need a cartesian join hash-one for rhs
      std::string rightTupleSetName = multiInputsComp->tupleSetNamesForInputs[rhsIndex];

      // the rhs column can be any column we only need it to get the number of rows
      std::vector<std::string> rightColumnsToApply = {rhsColumnNames[0]};

      // make the output tuple set
      mustache::mustache rightOutputTupleSetNameTemplate
          {"{{tupleSetMidTag}}_hashOneFor_{{RHSApplyAttribute}}_{{computationLabel}}_{{lambdaLabel}}_{{idx}}"};
      std::string rightOutputTupleSetName = rightOutputTupleSetNameTemplate.render(lambdaData);

      // make the column name
      mustache::mustache rightOutputColumnNameTemplate{"OneFor_right_{{computationLabel}}_{{lambdaLabel}}_{{idx}}"};
      std::string rightOutputColumnName = rightOutputColumnNameTemplate.render(lambdaData);

      // make the columns
      std::vector<std::string> rightOutputColumns = rhsColumnNames;
      rightOutputColumns.push_back(rightOutputColumnName);

      // make the lambda
      tcapString += formatLambdaComputation(rightTupleSetName,
                                            rhsColumnNames,
                                            rightColumnsToApply,
                                            rightOutputTupleSetName,
                                            rightOutputColumns,
                                            "HASHONE",
                                            computationNameWithLabel,
                                            "",
                                            {});


      /**
       * 1.3 Make the cartasian join
       */

      mustache::mustache
          outputTupleSetTemplate{"{{tupleSetMidTag}}_CartesianJoined__{{computationLabel}}_{{lambdaLabel}}_{{idx}}"};
      outputTupleSetName = outputTupleSetTemplate.render(lambdaData);

      // copy the output columns
      outputColumns = lhsColumnNames;
      outputColumns.insert(outputColumns.end(), rhsColumnNames.begin(), rhsColumnNames.end());

      // generate the join computation
      tcapString += formatJoinComputation(outputTupleSetName,
                                          outputColumns,
                                          leftOutputTupleSetName,
                                          {leftOutputColumnName},
                                          lhsColumnNames,
                                          rightOutputTupleSetName,
                                          {rightOutputColumnName},
                                          rhsColumnNames,
                                          computationNameWithLabel);

      // insert the string
      tcapStrings.emplace_back(std::move(tcapString));

      // go through each tuple set and update stuff
      for (int i = 0; i < multiInputsComp->tupleSetNamesForInputs.size(); ++i) {

        // check if this tuple set is the same index
        if (multiInputsComp->joinGroupForInput[i] == multiInputsComp->joinGroupForInput[lhsIndex] ||
            multiInputsComp->joinGroupForInput[i] == multiInputsComp->joinGroupForInput[rhsIndex]) {

          // the output tuple set is the new set with these columns
          multiInputsComp->tupleSetNamesForInputs[i] = outputTupleSetName;
          multiInputsComp->inputColumnsForInputs[i] = outputColumns;
          multiInputsComp->inputColumnsToApplyForInputs[i] = generatedColumns;

          // update the join group so that rhs has the same group as lhs
          multiInputsComp->joinGroupForInput[i] = multiInputsComp->joinGroupForInput[lhsIndex];
        }
      }

      // go to the next join
      idx++;
    }

  }

  /**
   * This method recursively goes through the entire lambda tree and generate the appropriate TCAP string for it
   *
   * @param computationLabel - the identifier of the computation
   * @param computationName - the name of the computation this lambda tree belongs to
   * @param isRootLambda - true if this is the root of a lambda tree false otherwise
   * @param multiInputsComp - the current input tuple sets
   * @param joinPredicate - is this a predicate or an expression
   * @param tcapStrings - the list of tcap strings generated
   */
  void getTCAPStrings(int computationLabel,
                      const std::string &computationName,
                      bool isRootLambda,
                      MultiInputsBase *multiInputsComp,
                      bool joinPredicate,
                      std::vector<std::string> &tcapStrings) {

    //  and empty parent lambda name means that this is the root of a lambda tree
    isRoot = isRootLambda;

    // set the computation label and name
    myComputationLabel = computationLabel;
    myComputationName = computationName;

    // make the name for this lambda object
    std::string myTypeName = this->getTypeOfLambda();
    std::string myName = myTypeName + "_" + std::to_string(myLambdaLabel + this->children.size());

    // should the child filter or not
    bool shouldFilterChild = false;
    if (myTypeName == "and" || myTypeName == "deref") {
      shouldFilterChild = joinPredicate;
    }

    // if this is an expressions subtree make sure all the inputs are joined
    if (isExpressionRoot) {

      // get all the inputs of this lambda tree
      std::set<int32_t> inputs;
      getAllInputs(inputs);

      // perform the cartasian joining if necessary
      generateJoinedInputs(tcapStrings, inputs, multiInputsComp);
    }

    // the names of the child lambda we need that for the equal lambda etc..
    std::vector<std::string> childrenLambdas;

    for (const auto &child : this->children) {

      // get the child lambda
      // if this is a predicated but the child is not the child is an expression root
      child.second->isExpressionRoot = joinPredicate && !shouldFilterChild;

      // recurse to generate the TCAP string
      child.second->getTCAPStrings(computationLabel,
                                   computationName,
                                   false,
                                   multiInputsComp,
                                   shouldFilterChild,
                                   tcapStrings);

      childrenLambdas.push_back(child.second->getLambdaName());
    }

    // generate the TCAP string for the current lambda
    std::string tcapString = this->generateTCAPString(multiInputsComp, joinPredicate);

    tcapStrings.push_back(tcapString);
  }

  /**
   * Generates the expression for the subtree of the lambda tree. We use this method usually to extract back the
   * columns missing after a join
   * @param prefix - the prefix added to the new tuple sets
   * @param multiInputsComp - the current input tuple sets
   * @param tcapStrings - the list of tcap strings generated
   */
  void generateExpressionTCAP(const std::string &prefix,
                              MultiInputsBase *multiInputsComp,
                              std::vector<std::string> &tcapStrings) {

    // it is not the root
    this->isRoot = false;

    // the prefix to be added to all atomic computations
    this->myPrefix = prefix;

    // the names of the child lambda we need that for the equal lambda etc..
    for (const auto &child : this->children) {

      // recurse to generate the TCAP string
      child.second->generateExpressionTCAP(prefix, multiInputsComp, tcapStrings);
    }

    // generate the TCAP string for the current lambda
    std::string tcapString = this->generateTCAPString(multiInputsComp, false);

    // store the tcap
    tcapStrings.push_back(tcapString);
  }

  virtual std::map<std::string, std::string> getInfo() = 0;

  /**
   * The children of this lambda that it consumes (the inputs that come from the other lambdas)
   */
  std::map<int32_t, LambdaObjectPtr> children;

  /**
   * The inputs of this lambda (the inputs that come from an external tuple set)
   */
  std::map<int32_t, uint32_t> inputIndexes;

  /**
   * Is the lambda object the root of the lambda tree
   */
  bool isRoot = false;

  /**
   * True if this is an expression subtree
   */
  bool isExpressionRoot = false;

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

  /**
   * Prefix we add to the tuple sets of this lambda
   */
  std::string myPrefix = "OutFor";

  /**
   * The name of the computations
   */
  std::string myComputationName;

  /**
   * The label associated with the computation
   */
  int32_t myComputationLabel;

  /**
   * The name of the lambda
   */
  int32_t myLambdaLabel;
};
}