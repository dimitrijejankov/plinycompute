#pragma once

#include "RHSShuffleJoinSource.h"

namespace pdb {

template<typename RHS>
class RHSJoinKeySource : public RHSShuffleJoinSource<RHS> {
public:

  RHSJoinKeySource(TupleSpec &inputSchema,
                   TupleSpec &hashSchema,
                   TupleSpec &recordSchema,
                   std::vector<int> &recordOrder,
                   const PDBAbstractPageSetPtr& rightInputPageSet,
                   uint64_t chunkSize) : RHSShuffleJoinSource<RHS>(inputSchema,
                                                                   rightInputPageSet,
                                                                   chunkSize) {

    // create the tuple set that we'll return during iteration
    this->output = std::make_shared<TupleSet>();

    // figure out the key att
    std::vector<int> matches = this->myMachine.match(hashSchema);
    this->keyAtt = matches[0];

    // figure the record attributes
    this->recordAttributes = this->myMachine.match(recordSchema);

    // allocate a vector for the columns
    this->columns = new void *[this->recordAttributes.size()];

    // create the columns for the records
    createCols<RHS>(this->columns, *this->output, 0, 0, recordOrder);

    // add the hash column
    this->output->addColumn(this->keyAtt, &this->hashColumn, false);

    PDBPageHandle page;
    while ((page = this->pageSet->getNextPage(0)) != nullptr) {

      // pin the page
      page->repin();

      // we grab the vector of hash maps
      Handle<JoinMap<RHS>> returnVal = ((Record<JoinMap<RHS>> *) (page->getBytes()))->getRootObject();

      // next we grab the join map we need
      this->maps.push_back(returnVal);

      // if the map has stuff add it to the queue
      auto it = this->maps.back()->begin();
      if (it != this->maps.back()->end()) {

        // insert the iterator
        this->pageIterators.push(it);

        // push the page
        this->pages.push_back(page);
      }
    }
  }

};

}