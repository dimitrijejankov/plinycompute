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

#ifndef COMP_LIST_CC
#define COMP_LIST_CC

#include <algorithm>
#include <AtomicComputationList.h>

#include "AtomicComputationList.h"
#include "AtomicComputationClasses.h"
#include "PDBDebug.h"

// gets the computation that builds the tuple set with the specified name
AtomicComputationPtr AtomicComputationList::getProducingAtomicComputation(std::string outputName) {
  if (producers.count(outputName) == 0) {
    PDB_COUT << "This could be bad... can't find the guy producing output " << outputName
             << ".\n";
  }
  return producers[outputName];
}

// gets the list of comptuations that consume the tuple set with the specified name
std::vector<AtomicComputationPtr> &AtomicComputationList::getConsumingAtomicComputations(std::string inputName) {
  if (consumers.count(inputName) == 0) {
    PDB_COUT << "This could be bad... can't find the guy consuming input " << inputName
             << ".\n";
  }
  return consumers[inputName];
}

// this effectively gets all of the leaves of the graph, since it returns all of the scans... every
// AtomicComputationPtr in the returned list will point to a SetScanner object
std::vector<AtomicComputationPtr> &AtomicComputationList::getAllScanSets() {
  return scans;
}

// add an atomic computation to the graph
void AtomicComputationList::addAtomicComputation(const AtomicComputationPtr& addMe) {

  if (addMe->getAtomicComputationType() == "Scan") {
    scans.push_back(addMe);
  }

  producers[addMe->getOutputName()] = addMe;
  if (consumers.count(addMe->getInputName()) == 0) {
    std::vector<AtomicComputationPtr> rhs;
    consumers[addMe->getInputName()] = rhs;
  }
  consumers[addMe->getInputName()].push_back(addMe);

  // now, see if this guy is a join; join is special, because we have to add both inputs to the
  // join to the consumers map
  if (addMe->getAtomicComputationTypeID() == ApplyJoinTypeID) {
    auto *myPtr = (ApplyJoin *) addMe.get();
    consumers[myPtr->getRightInput().getSetName()].push_back(addMe);
  } else if (addMe->getAtomicComputationTypeID() == UnionTypeID) {
    auto *myPtr = (Union *) addMe.get();
    consumers[myPtr->getRightInput().getSetName()].push_back(addMe);
  }

  // kill the copy of the shared pointer that is inside him
  addMe->destroyPtr();
}

std::ostream &operator<<(std::ostream &os, const AtomicComputationList &printMe) {
  for (auto &a : printMe.producers) {
      os << *a.second;
  }
  return os;
}

void AtomicComputationList::removeConsumer(const std::string &tupleSet, const AtomicComputationPtr& consumer) {

  // find the vector with consumers
  auto &cons = consumers[tupleSet];

  // remove
  cons.erase(std::remove(cons.begin(), cons.end(), consumer), cons.end());
}

void AtomicComputationList::removeProducer(const std::string &tupleSet) {
  producers.erase(tupleSet);
}

void AtomicComputationList::removeAllConsumers(const std::string &tupleSet) {

  // try to find it and remove it
  auto it = consumers.find(tupleSet);
  if(it != consumers.end()) {
    consumers.erase(it);
  }
}

void AtomicComputationList::replaceComputation(const std::string &tupleSet, const AtomicComputationPtr& comp) {

  // get the current comp that we want to replace
  auto &currComp = producers[tupleSet];

  /**
   * 0. Remove from the input the tuple set
   */

  // do we have an input
  if (!currComp->getInput().isEmpty()) {

    // replace the current comp
    auto &cons = consumers[currComp->getInput().getSetName()];
    cons.erase(std::remove(cons.begin(), cons.end(), currComp), cons.end());
  }

  // if we have two inputs update the other too
  if (currComp->hasTwoInputs()) {

    // replace the current comp
    auto &cons = consumers[currComp->getRightInput().getSetName()];
    cons.erase(std::remove(cons.begin(), cons.end(), currComp), cons.end());
  }

  /**
   * 1. Update the consumers so that they use the new tuple set
   */

  // for each of the consumers of this computation update input name
  for (auto &c : consumers[tupleSet]) {

    if (c->getInput().getSetName() == tupleSet) {

      // replace the set name
      c->getInput().setSetName(comp->getOutput().getSetName());
      c->getProjection().setSetName(comp->getOutput().getSetName());

    } else if (c->hasTwoInputs() && c->getRightInput().getSetName() == tupleSet) {

      // of this has to be the right input set the input
      c->getRightInput().setSetName(comp->getOutput().getSetName());
      c->getRightProjection().setSetName(comp->getOutput().getSetName());
    } else {
      throw std::runtime_error("Did not find the set name.");
    }
  }

  // rename the consumers
  auto tmp = consumers[tupleSet];
  consumers.erase(tupleSet);
  consumers[comp->getOutput().getSetName()] = tmp;

  /**
   * 2. Update the producer
   */

  // replace the producer
  producers.erase(tupleSet);
  producers[comp->getOutput().getSetName()] = comp;
}

void AtomicComputationList::removeAndRelink(const std::string &tupleSet) {

  // get the current comp that we want to replace
  auto &currComp = producers[tupleSet];

  // replace the producer
  producers.erase(tupleSet);

  // get the current consumers
  auto currCons = consumers[tupleSet];
  for(auto &c : currCons) {

    // check if we can rewire
    if(currComp->hasTwoInputs()) {
      throw std::runtime_error("Can not rewire computations with two inputs.");
    }

    // check if this is our computation
    if(c->getInput().getSetName() == currComp->getOutput().getSetName()) {

      // update the input to his one
      c->getInput().setSetName(currComp->getInput().getSetName());
      c->getProjection().setSetName(currComp->getProjection().getSetName());
    }

    // check if the right input is our computation
    if(c->hasTwoInputs() && c->getRightInput().getSetName() == currComp->getOutput().getSetName()) {

      // update the input to his one
      c->getRightInput().setSetName(currComp->getInput().getSetName());
      c->getRightProjection().setSetName(currComp->getProjection().getSetName());
    }
  }

  // update the consumers
  auto &inputCons = consumers[currComp->getInput().getSetName()];
  inputCons.erase(std::remove(inputCons.begin(), inputCons.end(), currComp), inputCons.end());
  inputCons.insert(inputCons.end(), currCons.begin(), currCons.end());
}

bool AtomicComputationList::hasConsumer(const std::string &tupleSet) {

  // try to find the consumers for this tuple set
  auto it = consumers.find(tupleSet);
  if (it == consumers.end()) {
    return false;
  }

  // check if it actually has consumers
  return !it->second.empty();
}


#endif
