#pragma once

#include <ComputeSink.h>
#include <TupleSpec.h>
#include <TupleSetMachine.h>
#include <JoinMap.h>
#include <JoinTuple.h>
#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlockMeta.h"

namespace pdb {

// this class is used to create a ComputeSink object that stores special objects that wrap up multiple columns of a tuple
template<typename RHSType>
class JoinKeySink : public ComputeSink {

 private:

  // tells us which attribute is the key
  int keyAtt;

  // if useTheseAtts[i] = j, it means that the i^th attribute that we need to extract from the input tuple is j
  std::vector<int> useTheseAtts;

  // if whereEveryoneGoes[i] = j, it means that the i^th entry in useTheseAtts goes in the j^th pos in the holder tuple
  std::vector<int> whereEveryoneGoes;

  // this is the list of columns that we are processing
  void **columns = nullptr;

  // the number of partitions
  size_t numPartitions;

 public:

  JoinKeySink(TupleSpec &inputSchema, TupleSpec &attsToOperateOn, TupleSpec &additionalAtts, std::vector<int> &whereEveryoneGoes, size_t numPartitions) : numPartitions(numPartitions), whereEveryoneGoes(whereEveryoneGoes) {

    // used to manage attributes and set up the output
    TupleSetSetupMachine myMachine(inputSchema);

    // figure out the key att
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    keyAtt = matches[0];

    // now, figure out the attributes that we need to store in the hash table
    useTheseAtts = myMachine.match(additionalAtts);
  }

  ~JoinKeySink() override {
    if (columns != nullptr)
      delete[] columns;
  }

  Handle<Object> createNewOutputContainer() override {

    // we simply create a new vector of join maps to store the output
    Handle<JoinMap<RHSType>> returnVal = makeObject<JoinMap<RHSType>>();
    returnVal->setHashValue(0);

    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {

    // get the map we are adding to
    Handle<JoinMap<RHSType>> myMap = unsafeCast<JoinMap<RHSType>>(writeToMe);

    // get all of the columns
    if (columns == nullptr)
      columns = new void *[whereEveryoneGoes.size()];

    int counter = 0;
    for (counter = 0; counter < whereEveryoneGoes.size(); counter++) {
      columns[counter] = (void *) &(input->getColumn<int>(useTheseAtts[whereEveryoneGoes[counter]]));
    }

    // this is where the hash attribute is located
    std::vector<size_t> &keyColumn = input->getColumn<size_t>(keyAtt);
    std::vector<Handle<pdb::matrix::MatrixBlockMeta>> &qwe = *(std::vector<Handle<pdb::matrix::MatrixBlockMeta>>*)columns[0];

    size_t length = keyColumn.size();
    for (int i = 0; i < length; i++) {


      // try to add the key... this will cause an allocation for a new key/val pair
      if (myMap->count(keyColumn[i]) == 0) {

        try {

          RHSType &temp = myMap->push(keyColumn[i]);
          pack(temp, i, 0, columns);

          // if we get an exception, then we could not fit a new key/value pair
        } catch (NotEnoughSpace &n) {

          // if we got here, then we ran out of space, and so we need to delete the already-processed
          // data so that we can try again...
          myMap->setUnused(keyColumn[i]);
          truncate<RHSType>(i, 0, columns);
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          throw n;
        }

        // the key is there
      } else {

        // and add the value
        RHSType *temp;
        try {

          temp = &(myMap->push(keyColumn[i]));

          // an exception means that we couldn't complete the addition
        } catch (NotEnoughSpace &n) {

          truncate<RHSType>(i, 0, columns);
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          throw n;
        }

        // now try to do the copy
        try {

          pack(*temp, i, 0, columns);

          // if the copy didn't work, pop the value off
        } catch (NotEnoughSpace &n) {

          myMap->setUnused(keyColumn[i]);
          truncate<RHSType>(i, 0, columns);
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          throw n;
        }
      }
    }
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("Join sink can not write out a page."); }

  // returns the number of records in the join key sink
  uint64_t getNumRecords(Handle<Object> &writeToMe) override {

    // get the map we were adding to
    Handle<JoinMap<RHSType>> myMap = unsafeCast<JoinMap<RHSType>>(writeToMe);

    // return the size
    return myMap->size();
  }
};

}