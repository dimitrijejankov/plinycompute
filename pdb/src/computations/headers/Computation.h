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

#include "Object.h"
#include "Lambda.h"
#include "ComputeSource.h"
#include "ComputeSink.h"
#include "PageProcessor.h"
#include "InputTupleSetSpecifier.h"
#include "PDBString.h"
#include "PDBAbstractPageSet.h"
#include <map>
#include <physicalAlgorithms/PDBSourcePageSetSpec.h>

namespace pdb {

// predefine the buffer manager interface
class PDBBufferManagerInterface;
using PDBBufferManagerInterfacePtr = std::shared_ptr<PDBBufferManagerInterface>;

// predefine the logical plan
struct LogicalPlan;
using LogicalPlanPtr = std::shared_ptr<LogicalPlan>;

// the compute plan
class ComputePlan;

// all nodes in a user-supplied computation are descended from this
class Computation : public Object {
 public:

  // if this particular computation can be used as a compute source in a pipeline, this
  // method will return the compute source object associated with the computation...
  //
  // In the general case, this method accepts the logical plan that this guy is a part of,
  // as well as the actual TupleSpec that this guy is supposed to produce, and then returns
  // a pointer to a ComputeSource object that can actually produce TupleSet objects corresponding
  // to that particular TupleSpec
  virtual ComputeSourcePtr getComputeSource(const PDBAbstractPageSetPtr &pageSet,
                                            size_t chunkSize,
                                            uint64_t workerID,
                                            std::map<ComputeInfoType, ComputeInfoPtr> &params) { return nullptr; }

  // likewise, if this particular computation can be used as a compute sink in a pipeline, this
  // method will return the compute sink object associated with the computation.  It requires the
  // TupleSpec that should be processed, as well as the projection of that TupleSpec that will
  // be put into the sink
  virtual ComputeSinkPtr getComputeSink(TupleSpec &consumeMe,
                                        TupleSpec &whichAttsToOpOn,
                                        TupleSpec &projection,
                                        uint64_t numberOfPartitions,
                                        std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                        pdb::LogicalPlanPtr &plan) { return nullptr; }

  virtual ComputeSinkPtr getKeySink(TupleSpec &consumeMe,
                                    TupleSpec &whichAttsToOpOn,
                                    TupleSpec &projection,
                                    uint64_t numberOfPartitions,
                                    std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                    pdb::LogicalPlanPtr &plan) { return nullptr; }

  // returns the type of this Computation
  virtual std::string getComputationType() = 0;

  //JiaNote: below function returns a TCAP string for this Computation
  virtual std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets, int computationLabel) = 0;

  // gets the name of the i^th input type...
  virtual std::string getInputType(int i) = 0;

  // check if the computation is keyed
  virtual bool isKeyed() = 0;

  // get the number of inputs to this query type
  virtual int getNumInputs() = 0;

  // get number of registered inputs
  int getNumOfRegisteredInputs(){
    return inputs == nullptr ? 0 : (int) inputs->size();
  }

  /**
   * gets the output type of this query as a string
   * @return
   */
  virtual std::string getOutputType() = 0;

  /**
   * set the first pos, by default
   * @param toMe
   * @return
   */
  bool setInput(const Handle<Computation>& toMe) {
    return setInput(0, toMe);
  }

  /**
   * sets the i^th input to be the output of a specific query... returns
   * true if this is OK, false if it is not
   * @param whichSlot
   * @param toMe
   * @return
   */
  bool setInput(int whichSlot, const Handle<Computation>& toMe) {

    // set the array of inputs if it is a nullptr
    if (inputs == nullptr) {
      inputs = makeObject<Vector<Handle<Computation>>>(getNumInputs());
      for (int i = 0; i < getNumInputs(); i++) {
        inputs->push_back(nullptr);
      }
    }

    // if we are adding this query to a valid pos
    if (whichSlot < getNumInputs()) {

      //make sure the output type of the guy we are accepting meets the input type
      if (getInputType(whichSlot) != toMe->getOutputType()) {
        std::cout << "Cannot set output of query node with output of type " << toMe->getOutputType()
                  << " to be the input";
        std::cout << " of a query with input type " << getInputType(whichSlot) << ".\n";
        return false;
      }
      (*inputs)[whichSlot] = toMe;

    } else {

      return false;
    }

    return true;
  }

  /**
   * This is implemented by the actual computation object... as the name implies, it is used
   * to extract the lambdas from the computation
   * @param returnVal
   */
  virtual void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) {}

  /**
   * to traverse from a graph sink recursively and generate TCAP
   * @param tcapStrings
   * @param computations
   * @param inputTupleSets
   * @param computationLabel
   */
  virtual void traverse(std::vector<std::string> &tcapStrings,
                        Vector<Handle<Computation>> &computations,
                        const std::vector<InputTupleSetSpecifier>& inputTupleSets,
                        int &computationLabel) {

    // so if the computation is not a source, go through each child computations and traverse them fist
    std::vector<InputTupleSetSpecifier> inputTupleSetsForMe;
    for (int i = 0; i < getNumOfRegisteredInputs() && !isSource; i++) {

      // get the child computation
      Handle<Computation> childComp = (*inputs)[i];

      // if we have not visited this computation visit it
      if (!childComp->traversed) {

        // go traverse the child computation
        childComp->traverse(tcapStrings, computations, inputTupleSets, computationLabel);

        // mark the computation as transversed
        childComp->traversed = true;

        // add the child computation
        computations.push_back(childComp);
      }

      // we met a computation that we have visited just grab the name of the output tuple set and the columns it has
      InputTupleSetSpecifier curOutput(childComp->outputTupleSetName, { childComp->outputColumnToApply }, { childComp->outputColumnToApply });
      inputTupleSetsForMe.push_back(curOutput);
    }

    // generate the TCAP string for this computation
    std::string curTCAPString = this->toTCAPString(inputTupleSetsForMe, computationLabel);

    // store the TCAP string generated
    tcapStrings.push_back(curTCAPString);

    // go to the next computation
    computationLabel++;
  }

  /**
   * Generates a SCAN for a page set, essentially we don't specify a database and set
   * @param computationLabel - the label of the computation the scan belongs to
   * @return the generated TCAP
   */
  std::string toSourceTCAP(int computationLabel) {


    // the template we are going to use to create the TCAP string for this ScanUserSet
    mustache::mustache scanSetTemplate{"inputDataFor{{computationType}}_{{computationLabel}}(in{{computationLabel}})"
                                       " <= SCAN ('{{computationType}}_{{computationLabel}}', [('pageSetID', '{{pageSetInteger}}:{{pageSetLabel}}')])\n"};

    mustache::data scanSetData;
    scanSetData.set("computationType", getComputationType());
    scanSetData.set("computationLabel", std::to_string(computationLabel));
    scanSetData.set("pageSetInteger", std::to_string(pageSetIdentifier.first));
    scanSetData.set("pageSetLabel", (std::string) pageSetIdentifier.second);

    // output column name
    mustache::mustache outputColumnNameTemplate{"in{{computationLabel}}"};

    //  set the output column name
    this->outputColumnToApply = outputColumnNameTemplate.render(scanSetData);
    std::vector<std::string> outputColumnNames = { this->outputColumnToApply };

    // output tuple set name template
    mustache::mustache outputTupleSetTemplate{"inputDataFor{{computationType}}_{{computationLabel}}"};
    this->outputTupleSetName = outputTupleSetTemplate.render(scanSetData);

    // update the state of the computation
    this->traversed = true;

    // return the TCAP string
    return std::move(scanSetTemplate.render(scanSetData));
  }

  /**
   * Clears the whole computation graph from any state it might have created during the traversal
   */
  void clearGraph() {

    // mark the we are not traversed
    traversed = false;

    // mark that we are not a source
    isSource = false;

    // clear all children
    for (int i = 0; i < getNumOfRegisteredInputs(); i++) {
      (*inputs)[i]->clearGraph();
    }
  }

  /**
   *
   * @param compID
   * @param pageSetName
   */
  void setPageSet(uint64_t compID, std::string &pageSetName) {

    // set the page set identifier
    pageSetIdentifier.first = compID;
    pageSetIdentifier.second = pageSetName;
  }

  /**
   * Mark this computation as a source
   */
  void markSource() {
    isSource = true;
  }

 protected:

  /**
   * The computations that are inputs to this computation
   */
  Handle<Vector<Handle<Computation>>> inputs = nullptr;

  /**
   * Did we traverse the computation
   */
  bool traversed = false;

  /**
   * Is this computation a source
   */
  bool isSource;

  /**
   * The last tuple set produced by this computation
   */
  String outputTupleSetName = "";

  /**
   * The column this computation is outputting that is going to be used by the next computation
   */
  String outputColumnToApply = "";

  /**
   * The name of the database the set we are scanning belongs to
   */
  pdb::String dbName;

  /**
   * The name of the set we are scanning
   */
  pdb::String setName;

  /**
   * The identifier of the page set
   */
  std::pair<size_t, pdb::String> pageSetIdentifier;
};

}