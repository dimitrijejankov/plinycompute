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

#ifndef JOB_H
#define JOB_H

#include "PDBString.h"
#include "Computation.h"
#include "Object.h"
#include "PDBVector.h"
#include "PageSetSpec.h"

// PRELOAD %Job%
// Question: what is the above line for?
namespace pdb {

/**
 * The Job class is the RequestType class which encapsulates everything about
 * a PhysicalAlgorithm that will be sent from the manager node to the worker nodes.
 */
class Job : public Object {
 private:
  int jobID; // Question: what type should this be?
  String tcapString;

  // Question: should these types by Handles or just the objects themselves?
  // I'm guessing that bc these classes are gonna be fairly nested, it's better
  // to have Handles here.

  // Each PageSetSpec contains information about a TCAP TupleSet (I think), a
  // ComputeSink/Source type, and a PageSet identifier.
  // Question: should we make a separate class for PageSetSpecs that have
  // Sink types, and PageSetSpecs that have Source types?
  //
  // Another question: should inputs and outputs be in physAlg? What makes more sense?
  Handle<Vector<Handle<PageSetSpec>>> inputs;
  Handle<PageSetSpec> output;
  Handle<PhysicalAlgorithm> physAlg;

  // The Computation objects that form this query graph.
  Handle<Vector<Handle<Computation>>> computations;
 public:
  // I'd say that because the Job is gonna be created by a call to one of the RequestFactory functions
  // (which call the ctor themselves), this is the only constructor we will need
  Job(int jobID,
      const String &tcapString,
      const Handle<Vector<Handle<PageSetSpec>>> &inputs,
      const Handle<PageSetSpec> &output,
      const Handle<PhysicalAlgorithm> &physAlg,
      const Handle<Vector<Handle<Computation>>> &computations,)
      : jobID(jobID),
        tcapString(tcapString),
        inputs(inputs),
        output(output),
        physAlg(physAlg),
        computations(computations) {}
  ENABLE_DEEP_COPY
};
}

#endif
