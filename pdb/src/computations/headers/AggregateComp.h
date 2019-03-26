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

#ifndef AGG_COMP
#define AGG_COMP

#include "Computation.h"
#include "HashSink.h"
#include "MapTupleSetIterator.h"
#include "DepartmentTotal.h"
#include "AggregateCompBase.h"

namespace pdb {

// this aggregates items of type InputClass.  To aggregate an item, the result of getKeyProjection () is
// used to extract a key from on input, and the result of getValueProjection () is used to extract a
// value from an input.  Then, all values having the same key are aggregated using the += operation over values.
// Note that keys must have operation == as well has hash () defined.  Also, note that values must have the
// + operation defined.
// 
// Once aggregation is completed, the key-value pairs are converted into OutputClass objects.  An object
// of type OutputClass must have two methods defined: KeyClass &getKey (), as well as ValueClass &getValue ().
// To convert a key-value pair into an OutputClass object, the result of getKey () is set to the desired key,
// and the result of getValue () is set to the desired value.
//
template<class OutputClass, class InputClass, class KeyClass, class ValueClass>
class AggregateComp : public AggregateCompBase {

  /**
   * Gets the operation that extracts a key from an input object
   * @param aggMe - the object we want to get the operation from
   * @return the projection lambda. This must be a lambda which takes in
   * a Handle<InputClass> and constructs an object of type KeyClass.
   */
  virtual Lambda<KeyClass> getKeyProjection(Handle<InputClass> aggMe) = 0;

  /**
   * Gets the operation that extracts a value from an input object
   * @param aggMe - the object we want to get the operation from
   * @return the projection lambda. This must be a lambda which takes in
   * a Handle<InputClass> and constructs an object of type ValueClass.
   */
  virtual Lambda<ValueClass> getValueProjection(Handle<InputClass> aggMe) = 0;



};

}

#endif
