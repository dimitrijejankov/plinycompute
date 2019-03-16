#include <utility>

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

#ifndef PIPELINE_H
#define PIPELINE_H

#include "ComputeSource.h"
#include "ComputeSink.h"
#include "ComputeExecutor.h"
#include "UseTemporaryAllocationBlock.h"
#include "Handle.h"
#include <queue>
#include <PDBAbstractPageSet.h>
#include <PDBAnonymousPageSet.h>

namespace pdb {

// this is used to buffer unwritten pages
struct MemoryHolder {
  // the output vector that this guy stores
  Handle<Object> outputSink;
  // page handle
  PDBPageHandle pageHandle;
  // the iteration where he was last written...
  // we use this because we cannot delete
  int iteration;
  void setIteration(int iterationIn);
  explicit MemoryHolder(const PDBPageHandle &pageHandle);
};

typedef std::shared_ptr<MemoryHolder> MemoryHolderPtr;

// this is a prototype for the pipeline
class Pipeline {

 private:

  // the id of the worker this pipeline is running on
  size_t workerID;

  // this is the page set where we are going to be writing all the output
  pdb::PDBAnonymousPageSetPtr outputPageSet;

  // this is the source of data in the pipeline
  ComputeSourcePtr dataSource;

  // this is where the pipeline goes to write the data
  ComputeSinkPtr dataSink;

  // here is our pipeline
  std::vector<ComputeExecutorPtr> pipeline;

  // and here is all of the pages we've not yet written back
  std::queue<MemoryHolderPtr> unwrittenPages;

 public:

  // the first argument is a function to call that gets a new output page...
  // the second argument is a function to call that deals with a full output page
  // the third argument is the iterator that will create TupleSets to process
  Pipeline(PDBAnonymousPageSetPtr outputPageSet, ComputeSourcePtr dataSource, ComputeSinkPtr tupleSink);

  ~Pipeline();

  // adds a stage to the pipeline
  void addStage(ComputeExecutorPtr addMe);

  // writes back any unwritten pages
  void cleanPages(int iteration);

  // runs the pipeline
  void run();

};

typedef std::shared_ptr<Pipeline> PipelinePtr;

}

#endif
