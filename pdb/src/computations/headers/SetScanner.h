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
#pragma once

#include "PDBAbstractPageSet.h"
#include "VectorTupleSetIterator.h"
#include "MapTupleSetIterator.h"
#include "SourceSetArg.h"
#include "Computation.h"
#include "PDBAggregationResultTest.h"
#include "JoinTupleTests.h"
#include "JoinArguments.h"
#include "Join8MainSource.h"

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
  std::string getInputType(int i) override {
    return "";
  }

  // for now scan is never keyed for now
  bool isKeyed() override {
    return pdb::tupleTests::has_get_key<OutputClass>::value;
  }

  // get the number of inputs to this query type
  int getNumInputs() override {
    return 0;
  }

  // gets the output type of this query as a string
  std::string getOutputType() override {
    return getTypeName<OutputClass>();
  }

  // below function implements the interface for parsing computation into a TCAP string
  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets,
                           int computationLabel) override {

    // the template we are going to use to create the TCAP string for this ScanUserSet
    mustache::mustache scanSetTemplate{"inputDataFor{{computationType}}_{{computationLabel}}(in{{computationLabel}})"
                                       " <= SCAN ('{{dbName}}', '{{setName}}', '{{computationType}}_{{computationLabel}}')\n"};

    // the data required to fill in the template
    mustache::data scanSetData;
    scanSetData.set("computationType", getComputationType());
    scanSetData.set("computationLabel", std::to_string(computationLabel));
    scanSetData.set("setName", std::string(setName));
    scanSetData.set("dbName", std::string(dbName));

    // output column name
    mustache::mustache outputColumnNameTemplate{"in{{computationLabel}}"};

    //  set the output column name
    std::string addedOutputColumnName = outputColumnNameTemplate.render(scanSetData);
    std::vector<std::string> outputColumnNames = { addedOutputColumnName };

    // output tuple set name template
    mustache::mustache outputTupleSetTemplate{"inputDataFor{{computationType}}_{{computationLabel}}"};
    std::string outputTupleSetName = outputTupleSetTemplate.render(scanSetData);

    // update the state of the computation
    this->traversed = true;
    this->outputTupleSetName = outputTupleSetName;
    this->outputColumnToApply = addedOutputColumnName;

    // return the TCAP string
    return scanSetTemplate.render(scanSetData);
  }

  /**
   * Returns the approprate source by returning the result of a proxy method @see _getComputeSource
   * @param pageSet - the page set we are scanning
   * @param chunkSize - the size of the chunk each tuple set is going to be
   * @param workerID - the id of the worker
   * @param params - the pipeline parameters we use to get the info about the set
   * @return can be either @see VectorTupleSetIterator or @see MapTupleSetIterator depending on the type of the set
   */
  pdb::ComputeSourcePtr getComputeSource(const PDBAbstractPageSetPtr &pageSet,
                                         uint64_t workerID,
                                         uint64_t numWorkers,
                                         std::map<ComputeInfoType, ComputeInfoPtr> &params) override {

    // get the join arguments
    JoinArgumentsPtr joinArgs = std::dynamic_pointer_cast<JoinArguments>(params[ComputeInfoType::JOIN_ARGS]);

    // if it is a join 8 create a join 8 main source iterator
    if(joinArgs->isJoin8) {
      return std::make_shared<pdb::Join8MainSource>(pageSet, workerID, numWorkers, *joinArgs->mappings, *joinArgs->joinedRecords);
    }

    // get the info about the source set
    auto sourceSetInfo = std::dynamic_pointer_cast<SourceSetArg>(params[ComputeInfoType::SOURCE_SET_INFO]);

    // check the type of the source set
    if(sourceSetInfo->set->containerType == PDB_CATALOG_SET_VECTOR_CONTAINER) {

      // return the vector set VectorTupleSetIterator
      return std::make_shared<pdb::VectorTupleSetIterator>(pageSet, workerID);
    }
    else if(sourceSetInfo->set->containerType == PDB_CATALOG_SET_MAP_CONTAINER) {

      // can we actually return a MapTupleSetIterator
      if constexpr(hasGetKey<OutputClass>::value and hasGetValue<OutputClass>::value) {

        // declare upfront the key and the value types
        using Value = typename std::remove_reference<decltype(std::declval<OutputClass>().getValue())>::type;
        using Key = typename std::remove_reference<decltype(std::declval<OutputClass>().getKey())>::type;

        // return the map iterator
        return std::make_shared<pdb::MapTupleSetIterator<typename remove_handle<Key>::type, typename remove_handle<Value>::type, OutputClass>> (pageSet, workerID);
      }
      else {

        // so for some reason bad stuff happened
        throw runtime_error("Can not get a MapTupleSetIterator for ("  + (string) dbName +"," + (string) setName +")");
      }
    }


    throw runtime_error("Unknown container  type for set ("  + (string) dbName +"," + (string) setName +")");
  }

  /**
   *
   */
  void traverse(std::vector<std::string> &tcapStrings,
                Vector<Handle<Computation>> &computations,
                const std::vector<InputTupleSetSpecifier>& inputTupleSets,
                int &computationLabel) override {

    // this is a scan set do stuff...
    if (!this->traversed) {

      std::string curTCAPString = this->toTCAPString(inputTupleSets, computationLabel);
      tcapStrings.push_back(curTCAPString);
      computationLabel++;
    }

    // get the output tuple set and the column
    std::string outputTupleSetName = this->outputTupleSetName;
    std::string addedOutputColumnName = this->outputColumnToApply;
    std::vector<std::string> outputColumnNames = { addedOutputColumnName };
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