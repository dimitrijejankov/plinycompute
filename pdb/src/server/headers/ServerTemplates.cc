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

#ifndef PDB_SERVER_TEMP_CC
#define PDB_SERVER_TEMP_CC

#include "Handle.h"
#include "PDBServer.h"
#include "ServerFunctionality.h"
#include <memory>

namespace pdb {

template<class Functionality>
void PDBServer::addFunctionality(std::shared_ptr<Functionality> functionality) {

  // first, get the name of this type
  std::string myType = getTypeName<Functionality>();

  // try to find one to check if it already exits
  auto it = functionalities.find(myType);
  if(it != functionalities.end()) {
    std::cerr << "BAD!  You can't add the same functionality twice.\n";
  }

  // add the functionality
  functionalities[myType] = functionality;

  // register the handlers
  functionality->recordServer(*this);
  functionality->recordComMgr(*connectionManager);
  functionality->init();
  functionality->registerHandlers(*this);
}

template <class Functionality>
Functionality& PDBServer::getFunctionality() {

  // and now, return the functionality
  return *getFunctionalityPtr<Functionality>();
}

template<class Functionality>
std::shared_ptr<Functionality> PDBServer::getFunctionalityPtr() {

  // first, get the name of this type
  std::string myType = getTypeName<Functionality>();

  // try to find the functionality
  auto it = functionalities.find(myType);
  if(it == functionalities.end()) {
    std::cerr << "BAD!  Could not find the functionality!\n";
  }

  // and now, return the functionality
  return std::dynamic_pointer_cast<Functionality>(it->second);
}


}

#endif
