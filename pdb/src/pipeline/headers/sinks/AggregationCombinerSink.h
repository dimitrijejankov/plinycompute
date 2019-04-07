//
// Created by dimitrije on 3/27/19.
//

#ifndef PDB_AGGREGATIONCOMBINERSINK_H
#define PDB_AGGREGATIONCOMBINERSINK_H

#include <ComputeSink.h>
#include <stdexcept>
#include <PDBPageHandle.h>

namespace pdb {

template<class KeyType, class ValueType>
class AggregationCombinerSink : public ComputeSink {
public:

  explicit AggregationCombinerSink(size_t workerID) : workerID(workerID) {}

  Handle<Object> createNewOutputContainer() override {

    // we simply create a new map to store the output
    Handle <Map <KeyType, ValueType>> returnVal = makeObject <Map <KeyType, ValueType>> ();
    return returnVal;
  }

  void writeOut(TupleSetPtr writeMe, Handle<Object> &writeToMe) override { throw std::runtime_error("AggregationCombinerSink can not write out tuple sets only pages."); }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override {

    // cast the hash table we are merging to
    Map<KeyType, ValueType> &mergeToMe = *unsafeCast <Map<KeyType, ValueType>> (writeToMe);

    // grab the hash table
    Handle<Object> hashTable = ((Record<Object> *) page->getBytes())->getRootObject();
    auto mergeMe = (*unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(hashTable))[workerID];

    // go through each key, value pair in the hash map we want to merge
    for(auto it = mergeMe->begin(); it != mergeMe->end(); ++it) {

      // if this key is not already there...
      if (mergeToMe.count ((*it).key) == 0) {

        // this point will record where the value is located
        ValueType *temp = nullptr;

        // try to add the key... this will cause an allocation for a new key/val pair
        try {
          // get the location that we need to write to...
          temp = &(mergeToMe[(*it).key]);

          // if we get an exception, then we could not fit a new key/value pair
        } catch (NotEnoughSpace &n) {

          // we do not deal with this, it must fit into a single hash table
          throw n;
        }

        // we were able to fit a new key/value pair, so copy over the value
        try {
          *temp = (*it).value;

          // if we could not fit the value...
        } catch (NotEnoughSpace &n) {

          // we do not deal with this, it must fit into a single hash table
          throw n;
        }

        // the key is there
      } else {

        // get the value and a copy of it
        ValueType &temp = mergeToMe[(*it).key];
        ValueType copy = temp;

        // and add to the old value, producing a new one
        try {
          temp = copy + (*it).value;

          // if we got here, then it means that we ram out of RAM when we were trying
          // to put the new value into the hash table
        } catch (NotEnoughSpace &n) {

          // we do not deal with this, it must fit into a single hash table
          throw n;
        }
      }
    }

  }

private:

  /**
   * The id of the worker
   */
  size_t workerID = 0;
};

}

#endif //PDB_AGGREGATIONCOMBINERSINK_H
