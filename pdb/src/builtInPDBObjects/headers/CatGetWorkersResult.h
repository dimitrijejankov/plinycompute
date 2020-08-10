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

#ifndef CATGETWORKERSRESULT_H
#define CATGETWORKERSRESULT_H

#include "Object.h"
#include "CatNodeObject.h"
#include "Handle.h"
#include "PDBString.h"
#include "PDBVector.h"
#include "PDBCatalogNode.h"

// PRELOAD %CatGetWorkersResult%

namespace pdb {

/**
 * Encapsulates a request to search for a type in the catalog
 */
class CatGetWorkersResult : public Object {

public:

  CatGetWorkersResult() = default;
  ~CatGetWorkersResult() = default;

  explicit CatGetWorkersResult(const std::vector<pdb::PDBCatalogNodePtr> &nodes) : nodes(pdb::makeObject<pdb::Vector<pdb::Handle<CatNodeObject>>>(nodes.size(), 0)) {

    // copy the stuff
    for(const auto &node : nodes) {
      this->nodes->push_back(pdb::makeObject<CatNodeObject>(node->address, node->port, node->nodeType, node->numCores, node->totalMemory, node->active));
    }
  }

  // Copy constructor
  explicit CatGetWorkersResult(const Handle<CatGetWorkersResult>& pdbItemToCopy) {
    nodes = pdbItemToCopy->nodes;
  }

  ENABLE_DEEP_COPY

  // the nodes that are workers
  pdb::Handle<pdb::Vector<pdb::Handle<CatNodeObject>>> nodes;
};
}

#endif
