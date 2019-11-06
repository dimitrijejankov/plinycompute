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

#pragma once

#include "ComputeSource.h"
#include "ComputeSink.h"
#include "ComputeExecutor.h"
#include "PageProcessor.h"
#include "UseTemporaryAllocationBlock.h"
#include "Handle.h"
#include <queue>
#include <PDBAbstractPageSet.h>
#include <PDBAnonymousPageSet.h>
#include <PipelineInterface.h>
#include <MemoryHolder.h>

namespace pdb {

// this is a prototype for the pipeline
class Pipeline : public PipelineInterface {

 private:

  // the id of the worker this pipeline is running on
  size_t workerID{};

  // this is the page set where we are going to be writing all the output
  pdb::PDBAnonymousPageSetPtr outputPageSet;

  // this is the source of data in the pipeline
  ComputeSourcePtr dataSource;

  // this is where the pipeline goes to write the data
  ComputeSinkPtr dataSink;

  // the page processor, we call this when we are have filled up a page, regardless of whether it contains stuff
  PageProcessorPtr pageProcessor;

  // here is our pipeline
  std::vector<ComputeExecutorPtr> pipeline;

  // and here is all of the pages we've not yet written back
  std::queue<MemoryHolderPtr> unwrittenPages;

  // this determines the size of the tuple set when running the pipeline
  PDBTupleSetSizePolicy tupleSetSizePolicy;

  // cleans the pipeline from all the leftover pages
  void cleanPipeline();

  // writes back any unwritten pages
  void cleanPages(int iteration);

 public:

  // the first argument is a function to call that gets a new output page...
  // the second argument is a function to call that deals with a full output page
  // the third argument is the iterator that will create TupleSets to process
  Pipeline(const PDBAnonymousPageSetPtr &outputPageSet, ComputeSourcePtr dataSource, ComputeSinkPtr tupleSink, PageProcessorPtr pageProcessor);

  ~Pipeline() override;

  // adds a stage to the pipeline
  void addStage(const ComputeExecutorPtr& addMe);

  // store page
  void addPageToIteration(const pdb::MemoryHolderPtr& ram, int iteration);

  // increment the record number1
  void incrementRecordNumber(const pdb::MemoryHolderPtr &ram);

  // runs the pipeline
  void run() override;

};

}
