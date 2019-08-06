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

#ifndef SELECTION_COMP
#define SELECTION_COMP

#include <sources/VectorTupleSetIterator.h>
#include <sinks/VectorSink.h>
#include "Computation.h"
#include "TypeName.h"

namespace pdb {

template<class OutputClass, class InputClass>
class SelectionComp : public Computation {

  // the computation returned by this method is called to see if a data item should be returned in the output set
  virtual Lambda<bool> getSelection(Handle<InputClass> checkMe) = 0;

  // the computation returned by this method is called to perfom a transformation on the input item before it
  // is inserted into the output set
  virtual Lambda<Handle<OutputClass>> getProjection(Handle<InputClass> checkMe) = 0;

  // calls getProjection and getSelection to extract the lambdas
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) override {
    Handle<InputClass> checkMe = nullptr;
    Lambda<bool> selectionLambda = getSelection(checkMe);
    Lambda<Handle<OutputClass>> projectionLambda = getProjection(checkMe);

    // the label we are started labeling
    int32_t startLabel = 0;

    // extract the lambdas
    selectionLambda.extractLambdas(returnVal, startLabel);
    projectionLambda.extractLambdas(returnVal, startLabel);
  }

  // this is a selection computation
  std::string getComputationType() override {
    return std::string("SelectionComp");
  }

  // gets the name of the i^th input type...
  std::string getInputType(int i) override {
    if (i == 0) {
      return getTypeName<InputClass>();
    } else {
      return "";
    }
  }

  pdb::ComputeSourcePtr getComputeSource(const PDBAbstractPageSetPtr &pageSet,
                                         size_t chunkSize,
                                         uint64_t workerID,
                                         std::map<ComputeInfoType, ComputeInfoPtr> &) override {
    return std::make_shared<pdb::VectorTupleSetIterator>(pageSet, chunkSize, workerID);
  }

  pdb::ComputeSinkPtr getComputeSink(TupleSpec &consumeMe, TupleSpec &, TupleSpec &projection, uint64_t,
                                     std::map<ComputeInfoType, ComputeInfoPtr> &, pdb::LogicalPlanPtr &) override {
    return std::make_shared<pdb::VectorSink<OutputClass>>(consumeMe, projection);
  }

  // get the number of inputs to this query type
  int getNumInputs() override {
    return 1;
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

    InputTupleSetSpecifier inputTupleSet = inputTupleSets[0];

    // make the inputs
    std::string inputTupleSetName = inputTupleSet.getTupleSetName();
    std::vector<std::string> inputColumnNames = inputTupleSet.getColumnNamesToKeep();
    std::vector<std::string> inputColumnsToApply = inputTupleSet.getColumnNamesToApply();

    /**
     * 1. Generate the TCAP for the selection predicate
     */

    // this is going to have info about the input
    assert(inputTupleSets.size() == 1);
    MultiInputsBase multiInputsBase(1);

    // set the name of the tuple set for the i-th position
    multiInputsBase.tupleSetNamesForInputs[0] = inputTupleSets[0].getTupleSetName();

    // set the columns for the i-th position
    multiInputsBase.inputColumnsForInputs[0] = inputTupleSets[0].getColumnNamesToKeep();

    // the the columns to apply for the i-th position
    multiInputsBase.inputColumnsToApplyForInputs[0] = inputTupleSets[0].getColumnNamesToApply();

    // setup all input names (the column name corresponding to input in tuple set) has to be one
    assert(inputTupleSets[0].getColumnNamesToApply().size() == 1);
    multiInputsBase.inputNames[0] = inputTupleSets[0].getColumnNamesToApply()[0];

    // we want to keep the input, so that it can be used by the projection
    multiInputsBase.inputColumnsToKeep = { inputTupleSets[0].getColumnNamesToKeep()[0] };

    // we label the lambdas within this computation from zero
    int lambdaLabel = 0;

    // call the selection
    GenericHandle checkMe (1);
    Lambda<bool> selectionLambda = getSelection(checkMe);

    std::string tcapString;
    tcapString += "\n/* Apply selection filtering */\n";
    tcapString += selectionLambda.toTCAPString(lambdaLabel,
                                               getComputationType(),
                                               computationLabel,
                                               false,
                                               &multiInputsBase);

    // get the columns for the TCAP
    auto appliedColumns = multiInputsBase.inputColumnsToApplyForInputs[0];
    auto outputColumns = multiInputsBase.getNotAppliedInputColumnsForIthInput(0);
    auto tupleSetName = multiInputsBase.tupleSetNamesForInputs[0];

    // make sure there is only one output column
    assert(outputColumns.size() == 1);

    // create the data for the column names
    mustache::data inputColumnsToApplyData = mustache::from_vector<std::string>(multiInputsBase.inputColumnsToApplyForInputs[0]);
    mustache::data outputColumnsData = mustache::from_vector<std::string>(multiInputsBase.getNotAppliedInputColumnsForIthInput(0));

    // create the data for the filter
    mustache::data selectionCompData;
    selectionCompData.set("computationType", getComputationType());
    selectionCompData.set("computationLabel", std::to_string(computationLabel));
    selectionCompData.set("inputColumns", inputColumnsToApplyData);
    selectionCompData.set("outputColumns", outputColumnsData);
    selectionCompData.set("tupleSetName", tupleSetName);

    // tupleSetName1(att1, att2, ...) <= FILTER (tupleSetName(methodCall_0OutFor_isFrank), methodCall_0OutFor_SelectionComp1(in0), 'SelectionComp_1')
    mustache::mustache scanSetTemplate
        {"filteredInputFor{{computationType}}{{computationLabel}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}) "
         "<= FILTER ({{tupleSetName}}({{#inputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/inputColumns}}), "
                    "{{tupleSetName}}({{#outputColumns}}{{value}}{{^isLast}},{{/isLast}}{{/outputColumns}}), '{{computationType}}_{{computationLabel}}')\n"};

    // generate the TCAP string for the FILTER
    tcapString += scanSetTemplate.render(selectionCompData);

    // template for the new tuple set name
    mustache::mustache newTupleSetNameTemplate{"filteredInputFor{{computationType}}{{computationLabel}}"};

    /**
     * 2. Generate the TCAP for the projection
     */

    // set the columns
    multiInputsBase.tupleSetNamesForInputs[0] = newTupleSetNameTemplate.render(selectionCompData);
    multiInputsBase.inputColumnsForInputs[0] = outputColumns;
    multiInputsBase.inputColumnsToApplyForInputs[0] = outputColumns;
    multiInputsBase.inputColumnsToKeep.clear();

    // get the projection
    Lambda<Handle<OutputClass>> projectionLambda = getProjection(checkMe);

    //TODO this needs to be made nicer
    std::string outputTupleSetName;
    std::vector<std::string> outputColumnNames;
    std::string addedOutputColumnName;
    std::string newTupleSetName;

    // generate the TCAP string for the FILTER
    tcapString += "\n/* Apply selection projection */\n";
    tcapString += projectionLambda.toTCAPString(lambdaLabel,
                                                getComputationType(),
                                                computationLabel,
                                                true,
                                                &multiInputsBase);
    tcapString += '\n';

    //  get the output columns
    outputColumns = multiInputsBase.inputColumnsToApplyForInputs[0];
    assert(outputColumns.size() == 1);
    this->outputColumnToApply = outputColumns[0];

    // update the tuple set
    this->outputTupleSetName = multiInputsBase.tupleSetNamesForInputs[0];

    // update the state of the computation
    this->traversed = true;

    // return the TCAP string
    return tcapString;
  }


};

}

#endif