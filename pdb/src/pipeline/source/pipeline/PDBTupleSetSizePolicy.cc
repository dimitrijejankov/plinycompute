#include <stdexcept>
#include <iostream>
#include <limits>
#include <pipeline/PDBTupleSetSizePolicy.h>
#include "PDBTupleSetSizePolicy.h"

namespace pdb {

PDBTupleSetSizePolicy::PDBTupleSetSizePolicy(uint64_t pageSize) : pageSize(pageSize) {}

void PDBTupleSetSizePolicy::pipelineFailed() {

  // if the chunk size is one we can no longer reduce it
  // therefore we have to have a pipeline failure
  if(chunkSize == 1) {
    throw std::runtime_error("We can not reduce the chunk size anymore so we fail here.");
  }

  // divide the chunk size by 2
  chunkSize /= 2;

  // mark that we have failed running the pipeline
  pipeline.initialFree = 0;
  pipeline.finalFree = 0;
  pipeline.numAdditionalPages = 0;
  pipeline.succeeded = false;

  // if we incremented last time disable increments we are at the edge on how much we can push the chunk size
  if(lastIncremented) {
    allowIncrements = false;
  }
}

void PDBTupleSetSizePolicy::pipelineSucceeded(uint64_t additionalPagesUsed, uint64_t initialFree, uint64_t finalFree) {

  // set the pipeline stats
  pipeline.initialFree = initialFree;
  pipeline.finalFree = finalFree;
  pipeline.numAdditionalPages = additionalPagesUsed;
  pipeline.succeeded = true;
}

void PDBTupleSetSizePolicy::writeToPageSucceeded(uint64_t additionalPages, uint64_t initialFree, uint64_t finalFree) {

  // set the write stats
  writeStats.initialFree = initialFree;
  writeStats.finalFree = finalFree;
  writeStats.numAdditionalPages = additionalPages;

  // track the maximum tuple size
  size_t numBytesUsed = ((int64_t)pipeline.initialFree - (int64_t) writeStats.finalFree) + (pipeline.numAdditionalPages + writeStats.numAdditionalPages) * pageSize;
  bytesAddedPerTuple = std::max(numBytesUsed / getChunksSize(), bytesAddedPerTuple);

  // figure out what the max is we can add
  if(bytesAddedPerTuple != 0) {

    // either cap it at 90% of the approximate number of tuples we can process, or 100 tuples
    maxChunkSize = std::min(90 * (pageSize / bytesAddedPerTuple) / 100 , 100ul);
  }
  else {

    // cap to 100 since we are not caring
    maxChunkSize = 100;
  }

  // if the pipeline succeeded we don't need to make it succeed
  if(pipeline.succeeded && pipeline.numAdditionalPages == 0 && additionalPages == 0) {
    tryToMakePipelineSucceed = false;
  }

  // if the pipeline succeeded and write succeeded without any additional pages then we
  // can possibly increase the chunk size
  if(allowIncrements && pipeline.succeeded && pipeline.numAdditionalPages == 0 && additionalPages == 0) {

    // so the rule is if we only used less than 75% of the page in total
    // increase chunk size 3 times because we expect the usage to go up to 60%
    auto percentage = ((pipeline.initialFree - writeStats.finalFree) * 100) / pageSize;

    // do the check and increment if it works
    if(percentage < 75) {
      chunkSize = std::min<uint64_t>(((uint64_t) chunkSize) * 3, maxChunkSize);
    }

    // just take the larger of two sizes where we succeeded
    lastFullySuccessfulSize = std::max(lastFullySuccessfulSize, chunkSize);

    // mark that we incremented in the previous iteration
    lastIncremented = true;

    // restart the retry
    pipelineRetry = 3;
  }
  else if(tryToMakePipelineSucceed && pipeline.numAdditionalPages != 0 && pipeline.succeeded && chunkSize != 1) {

    // if this is the first time we encountered this store it
    if(lastChunkSizeForWrite == -1) {
      lastChunkSizeForWrite = chunkSize;
    }

    // if we have used additional pages and the pipeline succeeded we want to disallow increments
    allowIncrements = false;

    // decrement the chunks size
    chunkSize /= 2;
  }
  // if we last incremented and we made the pipeline not succeed that means that we exceeded the
  // best chunk size therefore we need to revert to the last good known size.
  else if(lastIncremented && !tryToMakePipelineSucceed && pipeline.numAdditionalPages != 0 && pipeline.succeeded) {

    // revert back and don't mess with the increments
    chunkSize = lastFullySuccessfulSize;
    allowIncrements = false;
  }
  // if be got 3 pipeline failures in a row, it might be that we need to reduce the size down further
  else if(pipeline.numAdditionalPages != 0 && pipelineRetry == 0) {
    tryToMakePipelineSucceed = true;
    pipelineRetry = 3;
  }
  else if(pipeline.numAdditionalPages != 0) {
    pipelineRetry--;
  }

  // mark that we did not increment
  lastIncremented = false;
}

bool PDBTupleSetSizePolicy::inputWasProcessed() const {

  // if the pipeline succeeded we are fine, since not writing to the
  // output page is a critical failure
  return pipeline.succeeded;
}

int32_t PDBTupleSetSizePolicy::getChunksSize() const {
  return this->chunkSize;
}

bool PDBTupleSetSizePolicy::shouldGetNewPage(uint64_t freeLeft) const {

  // if the free memory is less than 75% of the estimated required memory grab a new page just to be sure...
  return  (3 * freeLeft) / 4 < chunkSize * bytesAddedPerTuple;
}

}