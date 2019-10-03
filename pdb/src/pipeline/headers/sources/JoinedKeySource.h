#pragma once

#include <JoinedShuffleJoinSource.h>


namespace pdb {

template<typename LHS>
class JoinedKeySource : public JoinedShuffleJoinSource<LHS> {
public:

  JoinedKeySource(TupleSpec &inputSchemaRHS,
                  TupleSpec &hashSchemaRHS,
                  TupleSpec &recordSchemaRHS,
                  const PDBAbstractPageSetPtr &lhsInputPageSet,
                  const std::vector<int> &lhsRecordOrder,
                  RHSShuffleJoinSourceBasePtr &rhsSource,
                  bool needToSwapLHSAndRhs,
                  uint64_t chunkSize) : JoinedShuffleJoinSource<LHS>(inputSchemaRHS,
                                                                     recordSchemaRHS,
                                                                     lhsRecordOrder,
                                                                     rhsSource,
                                                                     chunkSize) {

    PDBPageHandle page;
    while((page = lhsInputPageSet->getNextPage(0)) != nullptr) {

      // pin the page
      page->repin();

      // we grab the vector of hash maps
      Handle<JoinMap<LHS>> returnVal = ((Record<JoinMap<LHS>> *) (page->getBytes()))->getRootObject();

      // next we grab the join map we need
      this->lhsMaps.push_back(returnVal);

      if(this->lhsMaps.back()->size() != 0) {
        // insert the iterator
        this->lhsIterators.push(this->lhsMaps.back()->begin());

        // push the page
        this->lhsPages.push_back(page);
      }
    }

    // set up the output tuple
    this->output = std::make_shared<TupleSet>();
    this->lhsColumns = new void *[lhsRecordOrder.size()];

    // were the RHS and the LHS side swapped?
    if (!needToSwapLHSAndRhs) {

      // the right input will be put on offset-th column of the tuple set
      this->offset = (int) lhsRecordOrder.size();

      // the left input will be put at position 0
      createCols<LHS>(this->lhsColumns, *this->output, 0, 0, lhsRecordOrder);
    } else {

      // the right input will be put at the begining of the tuple set
      this->offset = 0;

      // the left input will be put at the recordOrder.size()-th column
      createCols<LHS>(this->lhsColumns, *this->output, (int) recordSchemaRHS.getAtts().size(), 0, lhsRecordOrder);
    }
  }

};

}
