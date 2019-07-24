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

#ifndef SET_WRITER_H
#define SET_WRITER_H

#include "VectorSink.h"
#include "Computation.h"
#include "TypeName.h"

namespace pdb {

template<class OutputClass>
class SetWriter : public Computation {

 public:

  SetWriter() = default;

  SetWriter(const String &dbName, const String &setName) : dbName(dbName), setName(setName) {}

  std::string getComputationType() override {
    return std::string("SetWriter");
  }

  // gets the name of the i^th input type...
  std::string getInputType(int i) override {
    if (i == 0) {
      return getTypeName<OutputClass>();
    } else {
      return "";
    }
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
  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets, int computationLabel) override {

    if (inputTupleSets.empty()) {
      return "";
    }

    InputTupleSetSpecifier inputTupleSet = inputTupleSets[0];
    return toTCAPString(inputTupleSet.getTupleSetName(),
                        inputTupleSet.getColumnNamesToKeep(),
                        inputTupleSet.getColumnNamesToApply(),
                        computationLabel);
  }

  std::string toTCAPString(std::string inputTupleSetName,
                           std::vector<std::string> &inputColumnNames,
                           std::vector<std::string> &inputColumnsToApply,
                           int computationLabel) {

    //Names for output stuff
    std::string outputTupleSetName = inputTupleSetName + "_out";
    std::vector<std::string> outputColumnNames = {""};
    std::string addedOutputColumnName;

    // the template we are going to use to create the TCAP string for this ScanUserSet
    mustache::mustache writeSetTemplate{"{{outputTupleSetName}}( {{outputColumnNames}}) <= "
                                        "OUTPUT ( {{inputTupleSetName}} ( {{inputColumnsToApply}} ), "
                                        "'{{dbName}}', '{{setName}}', '{{computationType}}_{{computationLabel}}')\n"};

    // the data required to fill in the template
    mustache::data writeSetData;
    writeSetData.set("outputTupleSetName", outputTupleSetName);
    writeSetData.set("outputColumnNames", outputColumnNames[0]);
    writeSetData.set("inputTupleSetName", inputTupleSetName);
    writeSetData.set("inputColumnsToApply", inputColumnsToApply[0]); //TODO? Only consider first column
    writeSetData.set("computationType", getComputationType());
    writeSetData.set("computationLabel", std::to_string(computationLabel));
    writeSetData.set("setName", std::string(setName));
    writeSetData.set("dbName", std::string(dbName));

    // update the state of the computation
    this->traversed = true;
    this->outputTupleSetName = outputTupleSetName;
    this->outputColumnToApply = addedOutputColumnName;

    // return the TCAP string
    return writeSetTemplate.render(writeSetData);
  }

  /**
   * Sets the output set of the set writer
   * @param dbName - the name of the database the set belongs to
   * @param setName - the name of the set
   */
  void setOutputSet(const std::string &db, const std::string &set) {

    // set the set identifier
    this->dbName = db;
    this->setName = set;
  }

  pdb::ComputeSinkPtr getComputeSink(TupleSpec &consumeMe, TupleSpec &, TupleSpec &projection, uint64_t ,
                                     std::map<ComputeInfoType, ComputeInfoPtr> &, pdb::LogicalPlanPtr &) override {
    return std::make_shared<pdb::VectorSink<OutputClass>>(consumeMe, projection);
  }

 private:

  /**
   * The name of the database the set we are scanning belongs to
   */
  pdb::String dbName;

  /**
   * The name of the set we are scanning
   */
  pdb::String setName;
};

}

#endif