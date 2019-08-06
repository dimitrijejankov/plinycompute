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

#ifndef AGG_COMP
#define AGG_COMP

#include "AggregateCompBase.h"
#include "MapTupleSetIterator.h"
#include "DepartmentTotal.h"
#include "PreaggregationSink.h"
#include "AggregationCombinerSink.h"

namespace pdb {

// this aggregates items of type InputClass.  To aggregate an item, the result of getKeyProjection () is
// used to extract a key from on input, and the result of getValueProjection () is used to extract a
// value from an input.  Then, all values having the same key are aggregated using the += operation over values.
// Note that keys must have operation == as well has hash () defined.  Also, note that values must have the
// + operation defined.
//
// Once aggregation is completed, the key-value pairs are converted into OutputClass objects.  An object
// of type OutputClass must have two methods defined: KeyClass &getKey (), as well as ValueClass &getValue ().
// To convert a key-value pair into an OutputClass object, the result of getKey () is set to the desired key,
// and the result of getValue () is set to the desired value.
//
template<class OutputClass, class InputClass, class KeyClass, class ValueClass>
class AggregateComp : public AggregateCompBase {

  // gets the operation tht extracts a key from an input object
  virtual Lambda<KeyClass> getKeyProjection(Handle<InputClass> aggMe) = 0;

  // gets the operation that extracts a value from an input object
  virtual Lambda<ValueClass> getValueProjection(Handle<InputClass> aggMe) = 0;

  // extract the key projection and value projection
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) override {
    Handle<InputClass> checkMe = nullptr;
    Lambda<KeyClass> keyLambda = getKeyProjection(checkMe);
    Lambda<ValueClass> valueLambda = getValueProjection(checkMe);

    // the label we are started labeling
    int32_t startLabel = 0;

    // extract the lambdas
    keyLambda.extractLambdas(returnVal, startLabel);
    valueLambda.extractLambdas(returnVal, startLabel);
  }

  // this is an aggregation comp
  std::string getComputationType() override {
    return std::string("AggregationComp");
  }

  int getNumInputs() override {
    return 1;
  }

  // gets the name of the i^th input type...
  std::string getInputType(int i) override {
    if (i == 0) {
      return getTypeName<InputClass>();
    } else {
      return "";
    }
  }

  // gets the output type of this query as a string
  std::string getOutputType() override {
    return getTypeName<OutputClass>();
  }

  // below function implements the interface for parsing computation into a TCAP string
  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets,
                           int computationLabel) override {

    if (inputTupleSets.empty()) {
      return "";
    }

    /**
     * 1. Generate the TCAP for the key extraction
     */

    // we label the lambdas from zero
    int lambdaLabel = 0;

    InputTupleSetSpecifier inputTupleSet = inputTupleSets[0];

    auto inputTupleSetName = inputTupleSet.getTupleSetName();
    auto inputColumnNames = inputTupleSet.getColumnNamesToKeep();
    auto inputColumnsToApply = inputTupleSet.getColumnNamesToApply();

    // this is going to have info about the inputs
    MultiInputsBase multiInputsBase(1);

    // set the name of the tuple set for the i-th position
    multiInputsBase.tupleSetNamesForInputs[0] = inputTupleSets[0].getTupleSetName();

    // set the columns for the i-th position
    multiInputsBase.inputColumnsForInputs[0] = inputTupleSets[0].getColumnNamesToKeep();

    // the the columns to apply for the i-th position
    multiInputsBase.inputColumnsToApplyForInputs[0] = inputTupleSets[0].getColumnNamesToApply();

    // setup all input names (the column name corresponding to input in tuple set)
    multiInputsBase.inputNames[0] = inputTupleSets[0].getColumnNamesToApply()[0];

    // we want to keep the input, so that it can be used by the projection
    multiInputsBase.inputColumnsToKeep = { inputTupleSets[0].getColumnNamesToKeep()[0] };

    //  get the projection lambda
    GenericHandle checkMe (1);
    Lambda<KeyClass> keyLambda = getKeyProjection(checkMe);

    std::string tcapString;
    tcapString += "\n/* Extract key for aggregation */\n";
    tcapString += keyLambda.toTCAPString(lambdaLabel,
                                         getComputationType(),
                                         computationLabel,
                                         false,
                                         &multiInputsBase);

    // get the key column
    assert(multiInputsBase.inputColumnsToApplyForInputs[0].size() == 1);
    auto keyColumn = multiInputsBase.inputColumnsToApplyForInputs[0][0];

    /**
     * 2. Generate the TCAP for the value extraction
     */

    // get the value lambda
    Lambda<ValueClass> valueLambda = getValueProjection(checkMe);

    // the the columns to apply for the i-th position
    multiInputsBase.inputColumnsToApplyForInputs[0] = inputTupleSets[0].getColumnNamesToApply();

    // we are ditching the input, keeping only the extracted value
    multiInputsBase.inputColumnsToKeep = { keyColumn };

    tcapString += "\n/* Extract value for aggregation */\n";
    tcapString += valueLambda.toTCAPString(lambdaLabel,
                                           getComputationType(),
                                           computationLabel,
                                           false,
                                           &multiInputsBase);

    // get the value column
    assert(multiInputsBase.inputColumnsToApplyForInputs[0].size() == 1);
    auto valueColumn = multiInputsBase.inputColumnsToApplyForInputs[0][0];

    /**
     * 3. Generate the TCAP for the aggregation
     */

    // set the output tuple set
    inputTupleSetName = multiInputsBase.tupleSetNamesForInputs[0];

    // create the data for the filter
    mustache::data clusterAggCompData;
    clusterAggCompData.set("computationType", getComputationType());
    clusterAggCompData.set("computationLabel", std::to_string(computationLabel));
    clusterAggCompData.set("inputTupleSetName", inputTupleSetName);

    // set the output columns
    mustache::mustache newAddedOutputColumnName1Template{"aggOutFor{{computationLabel}}"};
    std::string outputColumn = newAddedOutputColumnName1Template.render(clusterAggCompData);
    clusterAggCompData.set("outputColumn", outputColumn);

    // set the input columns
    mustache::data inputColumnsToApplyData = mustache::from_vector<std::string>({ keyColumn, valueColumn });
    clusterAggCompData.set("inputColumns", inputColumnsToApplyData);

    tcapString += "\n/* Apply aggregation */\n";
    mustache::mustache aggregateTemplate{"aggOutFor{{computationType}}{{computationLabel}} ({{outputColumn}})"
                                         "<= AGGREGATE ({{inputTupleSetName}}({{#inputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumns}}),"
                                         "'{{computationType}}_{{computationLabel}}')\n"};

    tcapString += aggregateTemplate.render(clusterAggCompData);

    // update the state of the computation
    mustache::mustache newTupleSetNameTemplate{"aggOutFor{{computationType}}{{computationLabel}}"};
    outputTupleSetName = newTupleSetNameTemplate.render(clusterAggCompData);

    // set the output column
    this->outputColumnToApply = outputColumn;

    // update marker
    this->traversed = true;

    return tcapString;
  }

  ComputeSinkPtr getComputeSink(TupleSpec &consumeMe, TupleSpec &, TupleSpec &projection, uint64_t numberOfPartitions,
                                std::map<ComputeInfoType, ComputeInfoPtr> &, pdb::LogicalPlanPtr &) override {
    return std::make_shared<pdb::PreaggregationSink<KeyClass, ValueClass>>(consumeMe, projection, numberOfPartitions);
  }

  ComputeSourcePtr getComputeSource(const PDBAbstractPageSetPtr &pageSet, size_t chunkSize, uint64_t workerID, std::map<ComputeInfoType, ComputeInfoPtr> &) override {
    return std::make_shared<pdb::MapTupleSetIterator<KeyClass, ValueClass, OutputClass>> (pageSet, workerID, chunkSize);
  }

  ComputeSinkPtr getAggregationHashMapCombiner(uint64_t workerID) override {
    return std::make_shared<pdb::AggregationCombinerSink<KeyClass, ValueClass>>(workerID);
  }

};

}

#endif
