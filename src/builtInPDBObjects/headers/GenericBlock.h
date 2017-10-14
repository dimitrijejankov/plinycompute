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
#ifndef GENERIC_BLOCK_H
#define GENERIC_BLOCK_H

// by Jia, Oct, 2016

#include "PDBVector.h"
#include "Object.h"

//  PRELOAD %GenericBlock%
namespace pdb {

// this class encapsulates a block of tuples/objects
// a page in user set can be transformed into a vector of generic blocks
// a generic block will be the basic unit of execution in pipeline
// most processors are based on generic block, except two: bundle processor and unbundle processor
// a bundle processor converts several pages into a vector of generic blocks;
// an unbundle processor converts a vector of generic blocks into several pages;

class GenericBlock : public Object {

private:
    Vector<Handle<Object>> block;

public:
    ENABLE_DEEP_COPY

    ~GenericBlock() {}

    GenericBlock() {}
    GenericBlock(size_t batchSize) : block(batchSize) {}
    Vector<Handle<Object>>& getBlock() {
        return block;
    }
};
}

#endif
