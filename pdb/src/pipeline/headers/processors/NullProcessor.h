//
// Created by dimitrije on 3/29/19.
//

#ifndef PDB_ZEROPROCESSOR_H
#define PDB_ZEROPROCESSOR_H

namespace pdb {

/**
 * This processor does not do anything with the page it simply returns true.
 * Meaning we always keep the page never discard it.
 */
class NullProcessor : public PageProcessor {

public:

  /**
   * Just returns true, does not processing
   * @param memory - the memory with the page and possibly the output sink, the output sink can be null
   * @return - true if we want to keep the page, false otherwise
   */
  bool process(const MemoryHolderPtr &memory) override {
    return true;
  }

};

}


#endif //PDB_ZEROPROCESSOR_H
