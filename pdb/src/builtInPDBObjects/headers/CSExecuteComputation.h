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

#ifndef EXEC_COMPUTATION_H
#define EXEC_COMPUTATION_H

#include "Object.h"
#include "PDBString.h"
#include <Computation.h>

// PRELOAD %CSExecuteComputation%

namespace pdb {

// encapsulates a request to run a query
class CSExecuteComputation : public Object {

public:

    CSExecuteComputation() = default;
    ~CSExecuteComputation() = default;

    CSExecuteComputation(Handle<Vector<Handle<Computation>>> &computations, const String &tcapString, size_t numBytes) {

      // set the num bytes
      this->numBytes = numBytes;

      // store the string
      this->tcapString = tcapString;

      // init the computations vector
      this->computations = makeObject<Vector<Handle<Computation>>>(computations->size(), 0);

      // copy the computations
      for(int i = 0; i < computations->size(); ++i) {
        this->computations->push_back((*computations)[i]);
      }
    }

    ENABLE_DEEP_COPY

    /**
     * The tcap string associated with the computations
     */
    String tcapString;

    /**
     * The computations
     */
    Handle<Vector<Handle<Computation>>> computations;

    /**
     * How large should the allocation block be to store the computations
     */
    size_t numBytes;
};
}

#endif
