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

#ifndef LAMBDA_HELPER_H
#define LAMBDA_HELPER_H

#include <memory>
#include <vector>
#include <functional>
#include "Object.h"
#include "Handle.h"
#include "Ptr.h"
#include "TupleSpec.h"
#include "ComputeExecutor.h"
#include "ComputeInfo.h"

namespace pdb {

// The basic idea is that we have a "class Lambda <typename Out>" that is returned by the query object.

// Internally, the query object creates a "class LambdaTree <Out>" object.  The reason that the
// query internally constructs a "class LambdaTree <Out>" whereas the query returns a 
// "class Lambda <Out>" is that there may be a mismatch between the two type parameters---the 
// LambdaTree may return a "class LambdaTree <Ptr<Out>>" object for efficiency.  Thus, we allow 
// a "class Lambda <Out>" object to be constructed with either a  "class LambdaTree <Out>"
// or a  "class LambdaTree <Ptr<Out>>".  We don't want to allow implicit conversions between
//  "class LambdaTree <Out>" and "class LambdaTree <Ptr<Out>>", however, which is why we need
// the separate type.

// Each "class LambdaTree <Out>" object is basically just a wrapper for a shared_ptr to a
// "TypedLambdaObject <Out> object".  So that we can pass around pointers to these things (without
// return types), "TypedLambdaObject <Out>" derives from "LambdaObject".

// forward delcaration
template<typename Out>
class TypedLambdaObject;

// we wrap up a shared pointer (rather than using a simple shared pointer) so that we 
// can override operations on these guys (if we used raw shared pointers, we could not)
template<typename ReturnType>
class LambdaTree {
protected:

  std::shared_ptr<TypedLambdaObject<ReturnType>> me;

public:

  LambdaTree() = default;

  LambdaTree(const LambdaTree<ReturnType> &toMe) : me(toMe.me) {}

  template<class Type>
  explicit LambdaTree(std::shared_ptr<Type> meIn) {
    me = meIn;
  }

  auto &getPtr() {
    return me;
  }

  LambdaTree<ReturnType> *operator->() const {
    return me.get();
  }

  LambdaTree<ReturnType> &operator*() const {
    return *me;
  }

  LambdaTree<ReturnType> &operator=(const LambdaTree<ReturnType> &toMe) {
    me = toMe.me;
    return *this;
  }

  template<class Type>
  LambdaTree<ReturnType> &operator=(std::shared_ptr<Type> toMe) {
    me = toMe;
    return *this;
  }

  unsigned int getInputIndex(int i) {
    return me->getInputIndex(i);
  }

};

}

#endif
