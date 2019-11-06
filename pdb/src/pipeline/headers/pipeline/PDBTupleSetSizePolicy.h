#pragma once

#include <cstdint>

namespace pdb {

class PDBTupleSetSizePolicy {
public:

  explicit PDBTupleSetSizePolicy(uint64_t pageSize);

  /**
   * This is supposed to be called to indicate that the pipeline failed while we were processing it.
   */
  void pipelineFailed();

  /**
   * This is supposed to be called
   * @param additionalPagesUsed - how many additional pages were used to process the pipeline
   * @param initialFree - how much did we have initially in the page we started with
   * @param finalFree - how much did we have in the page we ended with
   */
  void pipelineSucceeded(uint64_t additionalPagesUsed, uint64_t initialFree, uint64_t finalFree);

  /**
   * This is supposed to be called if a write to a page succeeded
   * @param initialFree - how much did we have initially in the page we started with
   * @param finalFree - how much did we have in the page we ended with
   */
  void writeToPageSucceeded(uint64_t additionalPages, uint64_t initialFree, uint64_t finalFree);

  /**
   * This will tell us if the input was processed or not. It will be used by the source to know if it should use the
   * the same input again
   * @return true if it was, false otherwise
   */
  [[nodiscard]] bool inputWasProcessed() const;

  /**
   * Returns the current chunks size we decided to use
   * @return the chunks size
   */
  [[nodiscard]] int32_t getChunksSize() const;

protected:

  /**
   * This is the size the policy has at the beginning of pipeline
   */
  int32_t chunkSize = 50;

  /**
   * Setting a max chunk size makes it so that we prevent, the chunk size from exploding.
   * And really making it too large does not yield any benefits the virtual function calls are amortized in this
   * case 1/maxChunkSize, if virtual functions calls are more expensive than processing 100 tuples that might
   * be problem with the computation.
   */
  int32_t maxChunkSize = 100;

  /**
   * The the last known chunk size we know the write succeeded.
   * This is used to restore back the the value of the chunk if we went too low.
   */
  int32_t lastChunkSizeForWrite = -1;

  /**
   * The last size that was fully successful
   */
  int32_t lastFullySuccessfulSize = -1;

  int32_t pipelineRetry = 3;

  // pipeline stats
  struct {

    // the initial free memory in the pipeline
    uint64_t initialFree{0};

    // the final free memory in the pipeline
    uint64_t finalFree{0};

    // the number of additional pages added
    uint64_t numAdditionalPages{0};

    // did we succeed in running the pipeline
    bool succeeded{true};

  } pipeline;

  // write stats
  struct {

    // the initial free memory in the pipeline
    uint64_t initialFree{0};

    // the final free memory in the pipeline
    uint64_t finalFree{0};

    // the number of additional pages added
    uint64_t numAdditionalPages{0};

  } writeStats;

  // did we increment the previous time
  bool lastIncremented = false;

  // do we still allow increments?
  // basically if we have an increment and after that increment there is a failure we disable the increments.
  bool allowIncrements = true;

  // in case the pipeline is using multiple pages to finish we want to decrease the chunk size until it can
  // finish without using additional pages since we need to redo expensive steps otherwise.
  bool tryToMakePipelineSucceed = true;

  // the size of the page
  uint64_t pageSize;
};

}
