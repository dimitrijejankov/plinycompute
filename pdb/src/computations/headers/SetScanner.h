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

#ifndef SCAN_SET_H
#define SCAN_SET_H

#include <PDBAbstractPageSet.h>
#include <sources/VectorTupleSetIterator.h>
#include "Computation.h"

namespace pdb {

template<class OutputClass>
class SetScanner : public Computation {
 public:

  SetScanner() = default;

  SetScanner(const std::string &db, const std::string &set) : dbName(db), setName(set) {}

  std::string getComputationType() override {
    return std::string("SetScanner");
  }

  // gets the name of the i^th input type...
  std::string getIthInputType(int i) override {
    return "";
  }

  // get the number of inputs to this query type
  int getNumInputs() override {
    return 0;
  }

  // gets the output type of this query as a string
  std::string getOutputType() override {
    return getTypeName<OutputClass>();
  }

  bool needsMaterializeOutput() override {
    return false;
  }

  // below function implements the interface for parsing computation into a TCAP string
  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets,
                           int computationLabel,
                           std::string &outputTupleSetName,
                           std::vector<std::string> &outputColumnNames,
                           std::string &addedOutputColumnName) override {

    InputTupleSetSpecifier inputTupleSet;
    if (!inputTupleSets.empty()) {
      inputTupleSet = inputTupleSets[0];
    }
    return toTCAPString(inputTupleSet.getTupleSetName(),
                        inputTupleSet.getColumnNamesToKeep(),
                        inputTupleSet.getColumnNamesToApply(),
                        computationLabel,
                        outputTupleSetName,
                        outputColumnNames,
                        addedOutputColumnName);
  }

  /**
   * Below function returns a TCAP string for this Computation
   * @param inputTupleSetName
   * @param inputColumnNames
   * @param inputColumnsToApply
   * @param computationLabel
   * @param outputTupleSetName
   * @param outputColumnNames
   * @param addedOutputColumnName
   * @return
   */
  std::string toTCAPString(std::string inputTupleSetName,
                           std::vector<std::string> &inputColumnNames,
                           std::vector<std::string> &inputColumnsToApply,
                           int computationLabel,
                           std::string &outputTupleSetName,
                           std::vector<std::string> &outputColumnNames,
                           std::string &addedOutputColumnName) {

    // the template we are going to use to create the TCAP string for this ScanUserSet
    mustache::mustache scanSetTemplate{"inputDataFor{{computationType}}_{{computationLabel}}(in{{computationLabel}})"
                                       " <= SCAN ('{{setName}}', '{{dbName}}', '{{computationType}}_{{computationLabel}}')\n"};

    // the data required to fill in the template
    mustache::data scanSetData;
    scanSetData.set("computationType", getComputationType());
    scanSetData.set("computationLabel", std::to_string(computationLabel));
    scanSetData.set("setName", std::string(setName));
    scanSetData.set("dbName", std::string(dbName));

    // output column name
    mustache::mustache outputColumnNameTemplate{"in{{computationLabel}}"};

    //  set the output column name
    addedOutputColumnName = outputColumnNameTemplate.render(scanSetData);
    outputColumnNames.push_back(addedOutputColumnName);

    // output tuple set name template
    mustache::mustache outputTupleSetTemplate{"inputDataFor{{computationType}}_{{computationLabel}}"};
    outputTupleSetName = outputTupleSetTemplate.render(scanSetData);

    // update the state of the computation
    this->setTraversed(true);
    this->setOutputTupleSetName(outputTupleSetName);
    this->setOutputColumnToApply(addedOutputColumnName);

    // return the TCAP string
    return scanSetTemplate.render(scanSetData);
  }

  pdb::ComputeSourcePtr getComputeSource(const PDBAbstractPageSetPtr &pageSet,
                                         size_t chunkSize,
                                         uint64_t workerID) override {
    return std::make_shared<pdb::VectorTupleSetIterator>(pageSet, chunkSize, workerID);
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