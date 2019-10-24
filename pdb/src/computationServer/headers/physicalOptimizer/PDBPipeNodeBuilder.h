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

#ifndef PDB_ADVANCEDPHYSICALNODEFACTORY_H
#define PDB_ADVANCEDPHYSICALNODEFACTORY_H

#include <set>
#include <AtomicComputation.h>
#include <cassert>
#include "PDBAbstractPhysicalNode.h"

namespace pdb {

class PDBPipeNodeBuilder {

public:

  PDBPipeNodeBuilder(size_t computationID,
                     std::unordered_map<uint64_t, bool> keyedComputations,
                     std::shared_ptr<AtomicComputationList> computations);

  /**
   *
   * @param sources
   * @return
   */
  std::vector<PDBAbstractPhysicalNodePtr> generateAnalyzerGraph();

 protected:

  /**
   *
   * @param curNode
   */
  void transverseTCAPGraph(const AtomicComputationPtr& curNode);

  /**
   * This method updates the @see consumedBy for the node we provide.
   * This method assumes that the last AtomicComputation belonging to this node is stored at @see currentPipe
   * @param node - the node we are updating the consumedBy for
   */
  void setConsumers(const std::shared_ptr<PDBAbstractPhysicalNode>& node);

  /**
   * After we create all the pipes we we need to connect them to create a graph consisting of pipes
   */
  void connectThePipes();

  /**
   * Does the pipeline supports the key only execution
   * @return true if it does, false otherwise
   */
  bool isCurrentPipeKeyed() {

    // go through each computation in the pipeline
    for(auto &c : currentPipe) {

      // figure out the computation ID
      std::string compName = c->getComputationName();
      uint64_t compID = std::stoi(compName.substr(compName.find('_') + 1));

      // if it is not
      if(!keyedComputations[compID]) {
        return false;
      }
    }

    // return true since all of them are keyed
    return true;
  }

  /**
   * This method creates a straight pipe and adds it to the physicalNodes
   */
  template <class T>
  void createPhysicalPipeline() {

    // this must never be empty
    assert(!currentPipe.empty());

    // create the node
    auto node = new T(currentPipe, computationID, currentNodeIndex++, isCurrentPipeKeyed());

    // create the node handle
    auto nodeHandle = node->getHandle();

    // update all the node connections
    setConsumers(nodeHandle);

    // is this a source node
    if(nodeHandle->hasScanSet()) {

      // add the source node
      physicalSourceNodes.push_back(nodeHandle);
    }

    // add the starts with
    startsWith[currentPipe.front()->getOutputName()].emplace_back(nodeHandle);

    // add the pipe to the physical nodes
    physicalNodes[nodeHandle->getNodeIdentifier()] = nodeHandle;

    // is this pipeline doing joining if it is store it in the join pipes
    if(nodeHandle->isJoining()) {
      joinPipes.push_back(nodeHandle);
    }
  }

  /**
   * The current node index
   */
  size_t currentNodeIndex;

  /**
   * All the nodes we already visited
   */
  std::set<AtomicComputationPtr> visitedNodes;

  /**
   * All the nodes that are in the current pipeline
   */
  std::vector<AtomicComputationPtr> currentPipe;

  /**
   * All the pipelines that start with a join
   */
  std::vector<PDBAbstractPhysicalNodePtr> joinPipes;

  /**
   * The physical nodes we created
   */
  std::map<std::string, PDBAbstractPhysicalNodePtr> physicalNodes;

  /**
   * Source physical nodes we created
   */
  std::vector<PDBAbstractPhysicalNodePtr> physicalSourceNodes;

  /**
   * Maps each pipe to the atomic computation it starts with.
   * The key is the name of the atomic computation the value is the pipe
   */
  std::map<std::string, std::vector<PDBAbstractPhysicalNodePtr>> startsWith;

  /**
   * Maps each pipe to the list of atomic computations that consume it
   */
  std::map<std::string, std::vector<std::string>> consumedBy;

  /**
   * All the source nodes we return them from the @see generateAnalyzerGraph
   */
  std::vector<PDBAbstractPhysicalNodePtr> sources;

  /**
   * The atomic computations we are splitting up
   */
  std::shared_ptr<AtomicComputationList> atomicComps;

  /**
   * Tells us what computations support key only computations
   */
  std::unordered_map<uint64_t, bool> keyedComputations;

  /**
   * The id of the computation we are building the pipes for
   */
  size_t computationID;
};

}



#endif //PDB_ADVANCEDPHYSICALNODEFACTORY_H
