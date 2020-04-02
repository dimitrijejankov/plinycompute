#include <utility>
#include <pipeline/Pipeline.h>


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

#include "Pipeline.h"
#include <utility>

pdb::Pipeline::Pipeline(const PDBAbstractPageSetPtr &outputPageSet,
                        ComputeSourcePtr dataSource,
                        ComputeSinkPtr tupleSink,
                        PageProcessorPtr pageProcessor) :
    tupleSetSizePolicy(outputPageSet->getMaxPageSize()),
    outputPageSet(outputPageSet),
    dataSource(std::move(dataSource)),
    dataSink(std::move(tupleSink)),
    pageProcessor(std::move(pageProcessor)) {}

pdb::Pipeline::~Pipeline() {

  // kill all of the pipeline stages
  while (!pipeline.empty())
    pipeline.pop_back();
}

// adds a stage to the pipeline
void pdb::Pipeline::addStage(const ComputeExecutorPtr& addMe) {
  pipeline.push_back(addMe);
}

// writes back any unwritten pages
void pdb::Pipeline::cleanPages(int iteration) {

  // take care of getting rid of any pages... but only get rid of those from two iterations ago...
  // pages from the last iteration may still have pointers into them
  while (!unwrittenPages.empty() && iteration > unwrittenPages.front()->iteration + 1) {

    // in this case, the page did not have any output data written to it... it only had
    // intermediate results, and so we will just discard it
    if (unwrittenPages.front()->outputSink == nullptr) {

      if (getNumObjectsInAllocatorBlock(unwrittenPages.front()->pageHandle->getBytes()) != 0) {

        // this is bad... there should not be any objects here because this memory
        // chunk does not store an output vector
        emptyOutContainingBlock(unwrittenPages.front()->pageHandle->getBytes());
      }

      // remove the page from the output set
      outputPageSet->removePage(unwrittenPages.front()->pageHandle);
      unwrittenPages.pop();

      // in this case, the page DID have some data written to it
    } else {

      // remove the page if not needed
      if(!pageProcessor->process(unwrittenPages.front())) {

        // and force the reference count for this guy to go to zero
        unwrittenPages.front()->outputSink.emptyOutContainingBlock();

        // remove the page from the page set
        outputPageSet->removePage(unwrittenPages.front()->pageHandle);
      }

      // OK, because we will have invalidated the current object allocator block, we need to
      // create a new one, or this could cause a lot of problems!!
      if (iteration == std::numeric_limits<int32_t>::max()) {
        makeObjectAllocatorBlock(1024, true);
      }

      // unpin the page so we don't have problems
      unwrittenPages.front()->pageHandle->unpin();

      // and get rid of him
      unwrittenPages.pop();
    }
  }
}

// cleans the pipeline
void pdb::Pipeline::cleanPipeline() {

  // finalize the sink
  dataSink->finalize();

  // first, reverse the queue so we go oldest to newest
  // this ensures that everything is deleted in the reverse order that it was created
  std::vector<MemoryHolderPtr> reverser;
  while (!unwrittenPages.empty()) {
    reverser.push_back(unwrittenPages.front());
    unwrittenPages.pop();
  }

  while (!reverser.empty()) {
    unwrittenPages.push(reverser.back());
    reverser.pop_back();
  }

  // write back all of the pages
  cleanPages(std::numeric_limits<int32_t>::max());

  if (!unwrittenPages.empty()) {
    std::cout << "This is bad: in destructor for pipeline, still some pages with objects!!\n";
  }

  // invalidate the current allocation block
  makeObjectAllocatorBlock(1024, true);
}

// runs the pipeline
void pdb::Pipeline::run() {

  // this is where we are outputting all of our results to
  MemoryHolderPtr ram = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

  // and here is the chunk
  TupleSetPtr curChunk;

  // the iteration counter
  int32_t iteration = 0;

  // we keep track of how much ram we used to process each iteration of the pipeline
  // this will be used by @see PDBTupleSetSizePolicy to determine the number of rows in the tuple set
  uint64_t initialFree = 0;
  uint64_t finalFree = 0;
  uint64_t additionalPagesUsed = 0;

  // while there is still data
  while ((curChunk = dataSource->getNextTupleSet(tupleSetSizePolicy)) != nullptr) {

    // we keep track of how much ram we used to process each iteration of the pipeline
    // this will be used by @see PDBTupleSetSizePolicy to determine the number of rows in the tuple set
    initialFree = getAllocator().getFreeBytesAtTheEnd();
    additionalPagesUsed = 0;

    // should we get a new page? we do this to try to avoid throwing an exception
    if(tupleSetSizePolicy.shouldGetNewPage(initialFree)) {

      // we add the current page to the list of output pages and then we grab a new one
      addPageToIteration(ram, iteration);
      ram = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

      // we just grabbed a new page
      initialFree = getAllocator().getFreeBytesAtTheEnd();
    }

    /**
     * 1. First we go through each computation in the pipeline and apply it
     *    what can happen is basically that stuff can not fit on a single page so we are going to get a bunch of pages
     *    that are connected somehow to do this computation.
     *    If the pipeline can not be processed we need to reduce the number of rows. This is going to be repeated
     */
    for(const auto &q : pipeline) {

      // this value indicates whether we need to reapply this computation
      bool reapply = false;

// this is kind of nasty but I am doing this so we can reapply a failed computation
// I am doing this since it is the easiest way to go and repeat this try block
REAPPLY:

      try {

        // try to process the chunk
        curChunk = q->process(curChunk);

      } catch (NotEnoughSpace &n) {

        //
        std::cout << "This is bad\n";

        // if we already reapplied then we can obviously not do the processing of this tuple set
        // we need to have less rows to finish this pipeline
        if(reapply) {

          // mark that we had a failure to process this pipeline
          tupleSetSizePolicy.pipelineFailed();

          // we add the current page to the list so it can be removed and then we grab a new one
          // the page will not contain anything important since we just grabbed it
          addPageToIteration(ram, iteration);
          ram = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

          // so in this case we need to reduce the number of rows by half to finish this pipeline
          goto CLEAN_ITERATION;
        }

        // we run out of space so this page can contain important data, process the page and possibly store it
        // the page can contain intermediate results
        addPageToIteration(ram, iteration);

        // get new page
        ram = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());
        additionalPagesUsed++;

        // jump to reapply and try to reprocess the chunk
        reapply = true;
        goto REAPPLY;
      }
    }

    // mark how much memory we have at the end in the last page we used
    finalFree = getAllocator().getFreeBytesAtTheEnd();

    // mark that we succeeded in running this pipeline
    tupleSetSizePolicy.pipelineSucceeded(additionalPagesUsed, initialFree, finalFree);

    // the initial size before we do the write to sink is how much free memory was at the end
    initialFree = finalFree;

    /**
     * 2. Write to the output pages and once we run out of memory process the page if needed.
     */

    // write the output pages
    try {

      // make a new output sink if we don't have one already
      if (ram->outputSink == nullptr) {
        ram->outputSink = dataSink->createNewOutputContainer();
      }

      // write the thing out
      dataSink->writeOut(curChunk, ram->outputSink);

    } catch (NotEnoughSpace &n) {

      std::cout << "This is bad\n";

      // increment the number of records if needed
      incrementRecordNumber(ram);

      // we need to keep the page
      addPageToIteration(ram, iteration);

      // get new page
      ram = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

      // and again, try to write back the output
      ram ->outputSink = dataSink->createNewOutputContainer();
      dataSink->writeOut(curChunk, ram->outputSink);
    }

    // mark how much memory we have at the end in the last page we used
    finalFree = getAllocator().getFreeBytesAtTheEnd();

    // mark that we succeeded in writing to a page
    tupleSetSizePolicy.writeToPageSucceeded(0, initialFree, finalFree);

// this is also nasty but basically if we have a tuple set that has too many rows
// we jump here to do some cleanup and repeat with a smaller chunk size
CLEAN_ITERATION:

    // lastly, write back all of the output pages
    iteration++;
    cleanPages(iteration);
  }

  // we need to keep the page
  addPageToIteration(ram, iteration);

  // for the last page increment the number of records
  incrementRecordNumber(ram);

  // clean the pipeline before we finish running
  cleanPipeline();
}

void pdb::Pipeline::addPageToIteration(const pdb::MemoryHolderPtr& ram, int iteration) {

  // set the iteration and store it in the list of unwritten pages
  ram->setIteration(iteration);
  unwrittenPages.push(ram);
}

void pdb::Pipeline::incrementRecordNumber(const pdb::MemoryHolderPtr &ram) {

  // if we have an output sink on this page update the number of records on the page set
  if (ram->outputSink != nullptr) {
    outputPageSet->increaseRecords(dataSink->getNumRecords(ram->outputSink));
  }
}