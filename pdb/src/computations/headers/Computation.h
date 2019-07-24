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

#ifndef COMPUTATION_H
#define COMPUTATION_H

#include "Object.h"
#include "Lambda.h"
#include "ComputeSource.h"
#include "ComputeSink.h"
#include "PageProcessor.h"
#include "InputTupleSetSpecifier.h"
#include "PDBString.h"
#include "PDBAbstractPageSet.h"
#include <map>

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

  // this is implemented by the actual computation object... as the name implies, it is used
  // to extract the lambdas from the computation
  virtual void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) {}

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

  // returns the type of this Computation
  virtual std::string getComputationType() = 0;

  //JiaNote: below function returns a TCAP string for this Computation
  virtual std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets,
                                   int computationLabel) = 0;

  // gets the name of the i^th input type...
  virtual std::string getIthInputType(int i) = 0;

  // get a handle to the i^th input to this query, which is also a query
  Handle<Computation> &getIthInput(int i) const {
    return (*inputs)[i];
  }

  // get the number of inputs to this query type
  virtual int getNumInputs() = 0;

  // gets the output type of this query as a string
  virtual std::string getOutputType() = 0;

  // set the first pos, by default
  bool setInput(const Handle<Computation>& toMe) {
    return setInput(0, toMe);
  }

  // sets the i^th input to be the output of a specific query... returns
  // true if this is OK, false if it is not
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
      if (getIthInputType(whichSlot) != toMe->getOutputType()) {
        std::cout << "Cannot set output of query node with output of type " << toMe->getOutputType()
                  << " to be the input";
        std::cout << " of a query with input type " << getIthInputType(whichSlot) << ".\n";
        return false;
      }
      (*inputs)[whichSlot] = toMe;

    } else {

      return false;
    }

    return true;
  }

  //whether the node has been traversed or not
  bool isTraversed() {
    return traversed;
  }

  //set the node to have been traversed
  void setTraversed(bool traversed) {

    this->traversed = traversed;
  }

  // to traverse from a graph sink recursively and generate TCAP
  virtual void traverse(std::vector<std::string> &tcapStrings,
                        const std::vector<InputTupleSetSpecifier>& inputTupleSets,
                        int &computationLabel) {

    // so if the computation is not a scan set, meaning it has at least one input process the children first
    // go through each child and traverse them
    std::vector<InputTupleSetSpecifier> inputTupleSetsForMe;
    for (int i = 0; i < this->getNumInputs(); i++) {

      // get the child computation
      Handle<Computation> childComp = (*inputs)[i];

      // if we have not visited this computation visit it
      if (!childComp->isTraversed()) {

        // go traverse the child computation
        childComp->traverse(tcapStrings, inputTupleSets, computationLabel);

        // mark the computation as transversed
        childComp->setTraversed(true);
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

protected:

  /**
   * The computations that are inputs to this computation
   */
  Handle<Vector<Handle<Computation>>> inputs = nullptr;

  /**
   *
   */
  bool traversed = false;

  /**
   *
   */
  String outputTupleSetName = "";

  /**
   *
   */
  String outputColumnToApply = "";

};

}

#endif
