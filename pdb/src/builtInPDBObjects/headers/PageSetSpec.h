//
// Created by vicram on 2/24/19.
//

#ifndef PDB_PAGESETSPEC_H
#define PDB_PAGESETSPEC_H

#include "PDBString.h"

namespace pdb {

/**
 * This is a planning-time construct. During query planning, PageSetSpec objects will
 * "point to" the inputs/outputs of PhysicalAlgorithms. This information will be sent
 * over the network, and worker nodes will use this to figure out how to construct
 * ComputeSources and ComputeSinks.
 *
 * Might split this into 2 classes on account of the Source/Sink split.
 */
class PageSetSpec : public Object {
 private:
  // This represents the name of a TupleSet in the TCAP string.
  // Question: will a PageSet always map exactly to a single TupleSet? Can a PageSet contain
  // only a part of a TupleSet? Can a PageSet contain more than one TupleSet?
  Handle<String> tupleSetID;

  // Identifies the type of ComputeSource or ComputeSink which can produce/consume this PageSet.
  // Can also have a null value which represents that this can correspond to any ComputeSource/Sink.
  // Question: is it actually useful to allow this to be null? I was under the impression that
  // the PhysicalOptimizer will decide which type of ComputeSource/Sink to use. But the PageSetSpec
  // is being created at the same time as the PhysicalOptimizer is making all those decisions, so wouldn't
  // we have full knowledge of which type of Source/Sink this is gonna be?
  //
  // Another question: should this be String or some type of enum?
  Handle<String> sourceOrSinkType;

  // Name of the PageSet which this will correspond to. Note that the PageSet is only created at
  // execution time, while the PageSetSpec (including pageSetID) is created at planning time.
  // This ID exists to uniquely label different PageSets at runtime.
  Handle<String> pageSetID;
};

}

#endif //PDB_PAGESETSPEC_H
