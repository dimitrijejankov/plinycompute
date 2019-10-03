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

#ifndef COMP_LIST_H
#define COMP_LIST_H

#include <iostream>
#include <memory>
#include <set>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "AtomicComputation.h"

// this is an indexed list of computations
struct AtomicComputationList {

private:
    // a map from the name of a TupleSet to the AtomicComputation that produced it
    std::map<std::string, AtomicComputationPtr> producers;

    // a map from the name of a TupleSet to the AtomicComputation(s) that will consume it
    std::map<std::string, std::vector<AtomicComputationPtr>> consumers;

    // a list of all of the SetScanner objects
    std::vector<AtomicComputationPtr> scans;

public:
    // gets the computation that builds the tuple set with the specified name
    AtomicComputationPtr getProducingAtomicComputation(std::string outputName);

    // gets the list of comptuations that consume the tuple set with the specified name
    std::vector<AtomicComputationPtr>& getConsumingAtomicComputations(std::string inputName);

    // this effectively gets all of the leaves of the graph, since it returns all of the scans...
    // every
    // AtomicComputationPtr in the returned list will point to a SetScanner object
    std::vector<AtomicComputationPtr>& getAllScanSets();

    // finds all the computations matching a predicate lambda
    template <class Predicate>
    std::set<AtomicComputationPtr> findByPredicate(Predicate p) {

      std::set<AtomicComputationPtr> ret;
      std::vector<AtomicComputationPtr> toVisit = scans;

      // visit all nodes and check the predicate
      while(!toVisit.empty()) {

        // get the node
        auto currNode = toVisit.back();
        toVisit.pop_back();

        // check the predicate
        if(p(currNode)) {
          ret.insert(currNode);
        }

        // get the current consumers and add them if you need it
        auto curCons = consumers.find(currNode->getOutputName());
        if(curCons != consumers.end()) {
          toVisit.insert(toVisit.end(), curCons->second.begin(), curCons->second.end());
        }
      }

      return std::move(ret);
    }

    // removes the consumer from a tuple set
    void removeConsumer(const std::string &tupleSet, const AtomicComputationPtr& consumer);

    // removes the atomic computation that produces the specified tuple set
    void removeProducer(const std::string &tupleSet);

    // remove all consumers
    void removeAllConsumers(const std::string &tupleSet);

    // removes all consumer entries that are not used
    void removeNonUsedConsumers();

    // removes the computation and relinks the produces of it and consumers of it
    void removeAndRelink(const std::string &tupleSet);

    // replace the computation
    void replaceComputation(const std::string &tupleSet, const AtomicComputationPtr& comp);

    // check if this tuple set has a consumer
    bool hasConsumer(const std::string &tupleSet);

    // add an atomic computation to the graph
    void addAtomicComputation(const AtomicComputationPtr& addMe);

    friend std::ostream& operator<<(std::ostream& os, const AtomicComputationList& printMe);
};

#endif
