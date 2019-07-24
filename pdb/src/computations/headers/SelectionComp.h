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
  virtual Lambda<bool> getSelection(Handle<InputClass> &checkMe) = 0;

  // the computation returned by this method is called to perfom a transformation on the input item before it
  // is inserted into the output set
  virtual Lambda<Handle<OutputClass>> getProjection(Handle<InputClass> &checkMe) = 0;

  // calls getProjection and getSelection to extract the lambdas
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) override {
    int suffix = 0;
    Handle<InputClass> checkMe = nullptr;
    Lambda<bool> selectionLambda = getSelection(checkMe);
    Lambda<Handle<OutputClass>> projectionLambda = getProjection(checkMe);
    selectionLambda.toMap(returnVal, suffix);
    projectionLambda.toMap(returnVal, suffix);
  }

  // this is a selection computation
  std::string getComputationType() override {
    return std::string("SelectionComp");
  }

  // gets the name of the i^th input type...
  std::string getIthInputType(int i) override {
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
    std::vector<std::string> childrenLambdaNames;
    std::string myLambdaName;
    return toTCAPString(inputTupleSet.getTupleSetName(),
                        inputTupleSet.getColumnNamesToKeep(),
                        inputTupleSet.getColumnNamesToApply(),
                        childrenLambdaNames,
                        computationLabel,
                        myLambdaName);
  }

  /**
   * to return Selection tcap string
   * @param inputTupleSetName
   * @param inputColumnNames
   * @param inputColumnsToApply
   * @param childrenLambdaNames
   * @param computationLabel
   * @param outputTupleSetName
   * @param outputColumnNames
   * @param addedOutputColumnName
   * @param myLambdaName
   * @return
   */
  std::string toTCAPString(std::string inputTupleSetName,
                           std::vector<std::string> &inputColumnNames,
                           std::vector<std::string> &inputColumnsToApply,
                           std::vector<std::string> &childrenLambdaNames,
                           int computationLabel,
                           std::string &myLambdaName) {

    PDB_COUT << "ABOUT TO GET TCAP STRING FOR SELECTION" << std::endl;
    Handle<InputClass> checkMe = nullptr;
    std::string tupleSetName;
    std::vector<std::string> columnNames;
    std::string addedColumnName;
    int lambdaLabel = 0;

    PDB_COUT << "ABOUT TO GET TCAP STRING FOR SELECTION LAMBDA" << std::endl;
    Lambda<bool> selectionLambda = getSelection(checkMe);

    std::string tcapString;
    tcapString += "\n/* Apply selection filtering */\n";
    tcapString += selectionLambda.toTCAPString(inputTupleSetName,
                                               inputColumnNames,
                                               inputColumnsToApply,
                                               childrenLambdaNames,
                                               lambdaLabel,
                                               getComputationType(),
                                               computationLabel,
                                               tupleSetName,
                                               columnNames,
                                               addedColumnName,
                                               myLambdaName,
                                               false);

    PDB_COUT << "The tcapString after parsing selection lambda: " << tcapString << "\n";
    PDB_COUT << "lambdaLabel=" << lambdaLabel << "\n";

    // create the data for the column names
    mustache::data inputColumnData = mustache::data::type::list;
    for (int i = 0; i < inputColumnNames.size(); i++) {

      mustache::data columnData;

      // fill in the column data
      columnData.set("columnName", inputColumnNames[i]);
      columnData.set("isLast", i == inputColumnNames.size() - 1);

      inputColumnData.push_back(columnData);
    }

    // create the data for the filter
    mustache::data selectionCompData;
    selectionCompData.set("computationType", getComputationType());
    selectionCompData.set("computationLabel", std::to_string(computationLabel));
    selectionCompData.set("inputColumns", inputColumnData);
    selectionCompData.set("tupleSetName", tupleSetName);
    selectionCompData.set("addedColumnName", addedColumnName);

    // tupleSetName1(att1, att2, ...) <= FILTER (tupleSetName(methodCall_0OutFor_isFrank), methodCall_0OutFor_SelectionComp1(in0), 'SelectionComp_1')
    mustache::mustache scanSetTemplate
        {"filteredInputFor{{computationType}}{{computationLabel}}({{#inputColumns}}{{columnName}}{{^isLast}}, {{/isLast}}{{/inputColumns}}) "
         "<= FILTER ({{tupleSetName}}({{addedColumnName}}), {{tupleSetName}}({{#inputColumns}}{{columnName}}{{^isLast}}, {{/isLast}}{{/inputColumns}}), '{{computationType}}_{{computationLabel}}')\n"};

    // generate the TCAP string for the FILTER
    tcapString += scanSetTemplate.render(selectionCompData);

    // template for the new tuple set name
    mustache::mustache newTupleSetNameTemplate{"filteredInputFor{{computationType}}{{computationLabel}}"};

    // generate the new tuple set name
    std::string newTupleSetName = newTupleSetNameTemplate.render(selectionCompData);

    PDB_COUT << "TO GET TCAP STRING FOR PROJECTION LAMBDA\n";
    Lambda<Handle<OutputClass>> projectionLambda = getProjection(checkMe);

    //TODO this needs to be made nicer
    std::string outputTupleSetName;
    std::vector<std::string> outputColumnNames;
    std::string addedOutputColumnName;

    // generate the TCAP string for the FILTER
    tcapString += "\n/* Apply selection projection */\n";
    tcapString += projectionLambda.toTCAPString(newTupleSetName,
                                                inputColumnNames,
                                                inputColumnsToApply,
                                                childrenLambdaNames,
                                                lambdaLabel,
                                                getComputationType(),
                                                computationLabel,
                                                outputTupleSetName,
                                                outputColumnNames,
                                                addedOutputColumnName,
                                                myLambdaName,
                                                true);

    // update the state of the computation
    this->setTraversed(true);
    this->outputTupleSetName = outputTupleSetName;
    this->outputColumnToApply = addedOutputColumnName;

    // return the TCAP string
    return tcapString;
  }


};

}

#endif