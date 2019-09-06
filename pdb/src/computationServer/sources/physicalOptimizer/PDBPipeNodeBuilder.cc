#include <utility>

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

#include <physicalOptimizer/PDBJoinPhysicalNode.h>
#include "physicalOptimizer/PDBPipeNodeBuilder.h"
#include "physicalOptimizer/PDBStraightPhysicalNode.h"
#include "physicalOptimizer/PDBAggregationPhysicalNode.h"
#include "AtomicComputationList.h"


namespace pdb {

PDBPipeNodeBuilder::PDBPipeNodeBuilder(size_t computationID, std::shared_ptr<AtomicComputationList> computations)
    : atomicComps(std::move(computations)), currentNodeIndex(0), computationID(computationID) {}
}

std::vector<pdb::PDBAbstractPhysicalNodePtr> pdb::PDBPipeNodeBuilder::generateAnalyzerGraph() {

  // go through each source in the sources
  for(const AtomicComputationPtr &source : atomicComps->getAllScanSets()) {

    // go trough each consumer of this node
    for(const auto &consumer : atomicComps->getConsumingAtomicComputations(source->getOutputName())) {

      // we start with a source so we push that back
      currentPipe.push_back(source);

      // add the consumer to the pipe
      currentPipe.push_back(consumer);

      // then we start transversing the graph upwards
      transverseTCAPGraph(consumer);
    }
  }

  // connect the pipes
  connectThePipes();

  // return the generated source nodes
  return this->physicalSourceNodes;
}

void pdb::PDBPipeNodeBuilder::transverseTCAPGraph(AtomicComputationPtr curNode) {

  // did we already visit this node
  if(visitedNodes.find(curNode) != visitedNodes.end()) {

    // pop this operator that we are currently on since we have visited it
    currentPipe.pop_back();

    // if there is something in the pipeline materialize it with a straight pipe
    if(!currentPipe.empty()) {
      createPhysicalPipeline<pdb::PDBStraightPhysicalNode>();
    }

    // clear the pipe we are done here
    currentPipe.clear();

    // we are done here
    return;
  }

  // ok now we visited this node
  visitedNodes.insert(curNode);

  // check the type of this node might be a pipeline breaker
  switch (curNode->getAtomicComputationTypeID()) {

    case HashOneTypeID:
    case HashLeftTypeID:
    case HashRightTypeID: {

      // we got a hash operation, create a PDBJoinPhysicalNode
      createPhysicalPipeline<PDBJoinPhysicalNode>();
      currentPipe.clear();

      break;
    }
    case ApplyAggTypeID: {

      // we got a aggregation so we need to create an PDBAggregationPhysicalNode
      // we need to remove the ApplyAgg since it will be run in the next pipeline this one is just preparing the data for it.
      currentPipe.pop_back();
      createPhysicalPipeline<PDBAggregationPhysicalNode>();

      // add the ApplyAgg back to an empty pipeline
      currentPipe.clear();
      currentPipe.push_back(curNode);

      break;
    }
    case UnionTypeID: {

      // we are a side so we need to create a straight pipeline and remove the Union because it is going to be executed
      // in the next pipeline
      currentPipe.pop_back();
      createPhysicalPipeline<PDBStraightPhysicalNode>();

      // add the Union back to an empty pipeline
      currentPipe.clear();
      currentPipe.push_back(curNode);

      break;
    }
    case WriteSetTypeID: {

      // write set also breaks the pipe because this is where the pipe ends
      createPhysicalPipeline<pdb::PDBStraightPhysicalNode>();
      currentPipe.clear();
    }
    default: {

      // we only care about these since they tend to be pipeline breakers
      break;
    }
  }

  // grab all the consumers
  auto consumers = atomicComps->getConsumingAtomicComputations(curNode->getOutputName());

  // in the case that we have some multiple consumers and we might want to move stuff over from this pipeline
  // to the consumer pipelines for example if we only have one apply aggregation, which would not do anything
  std::vector<AtomicComputationPtr> moveTheseOver;

  // if we have multiple consumers and there is still stuff left in the pipe
  if(consumers.size() > 1 && !currentPipe.empty()) {

    // move the last computation to the next pipeline since it needs to start from it
    moveTheseOver.emplace_back(currentPipe.back());

    // in the case that we only have one ApplyAgg that is going to be moved to the next pipeline just ignore it.
    if(currentPipe.size() == 1 && currentPipe.front()->getAtomicComputationTypeID() == ApplyAggTypeID) {
      currentPipe.clear();
    }
    else {

      // otherwise this is a pipeline breaker create a pipe
      createPhysicalPipeline<PDBStraightPhysicalNode>();
      currentPipe.clear();
    }
  }

  // go through each consumer and transverse to get the next pipe
  for(auto &consumer : consumers) {
    currentPipe.insert(currentPipe.end(), moveTheseOver.begin(), moveTheseOver.end());
    currentPipe.push_back(consumer);
    transverseTCAPGraph(consumer);
  }
}

void pdb::PDBPipeNodeBuilder::setConsumers(std::shared_ptr<PDBAbstractPhysicalNode> node) {

  // all the consumers of these pipes
  std::vector<std::string> consumers;

  // go trough each consumer of this node
  auto &consumingAtomicComputations = atomicComps->getConsumingAtomicComputations(this->currentPipe.back()->getOutputName());

  // if we are only having one consumer then then next pipe starts with the atomic computation that consumes this one
  if(consumingAtomicComputations.size() == 1) {

    // add them to the consumers
    consumers.push_back(consumingAtomicComputations.front()->getOutputName());
    this->consumedBy[node->getNodeIdentifier()] = consumers;
  }
  else if(consumingAtomicComputations.size() > 1) {
    // add them to the consumers
    consumers.push_back(this->currentPipe.back()->getOutputName());
    this->consumedBy[node->getNodeIdentifier()] = consumers;
  }
}

void pdb::PDBPipeNodeBuilder::connectThePipes() {

  // connect all the consumers and producers
  for(const auto &node : physicalNodes) {

    // get all the consumers of this pipe
    auto consumingAtomicComputation = consumedBy[node.second->getNodeIdentifier()];

    // go through each at
    for(const auto &atomicComputation : consumingAtomicComputation) {

      // get the consuming pipeline
      auto consumers = startsWith[atomicComputation];

      for(const auto &consumer : consumers) {

        // add the consuming node of this guy
        node.second->addConsumer(consumer);
      }
    }
  }

  // go through each join pipe and connect the left and right side of the join so that they know about each other
  for(const auto &joinNode : joinPipes) {

    // get the producers
    auto &producers = joinNode->getProducers();
    if(producers.size() != 2) {
      throw std::runtime_error("There are not exactly two sides of a join pipe, something went wrong!");
    }

    // grab both sides of the join
    auto joinSide1 = producers.front();
    auto joinSide2 = producers.back();

    // set the other join side
    ((PDBJoinPhysicalNode*) joinSide1.get())->otherSide = joinSide2;
    ((PDBJoinPhysicalNode*) joinSide2.get())->otherSide = joinSide1;
  }
}