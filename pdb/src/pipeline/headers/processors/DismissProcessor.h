#pragma once

namespace pdb {

/**
 * This processor does not do anything with the page it simply returns false.
 * Meaning we always dismiss the page
 */
class DismissProcessor : public PageProcessor {

 public:

  /**
   * Just returns false, does not processing
   * @param memory - the memory with the page and possibly the output sink, the output sink can be null
   * @return - true if we want to keep the page, false otherwise
   */
  bool process(const MemoryHolderPtr &memory) override {
    return false;
  }

};

}