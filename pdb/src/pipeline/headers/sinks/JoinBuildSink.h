//
// Created by dimitrije on 2/19/19.
//

#pragma once

#include <ComputeSink.h>
#include <TupleSpec.h>
#include <TupleSetMachine.h>
#include <JoinMap.h>
#include <JoinTuple.h>

namespace pdb {

// this class is used to create a ComputeSink object that stores special objects that wrap up multiple columns of a tuple
template<typename RHSType>
class JoinBuildSink : public ComputeSink {

private:

  // tells us which attribute is the key
  int keyAtt;

  // if useTheseAtts[i] = j, it means that the i^th attribute that we need to extract from the input tuple is j
  std::vector<int> useTheseAtts;

  // if whereEveryoneGoes[i] = j, it means that the i^th entry in useTheseAtts goes in the j^th slot in the holder tuple
  std::vector<int> whereEveryoneGoes;

  // this is the list of columns that we are processing
  void **columns = nullptr;

  // the number of partitions
  size_t numPartitions;

public:

  JoinBuildSink(TupleSpec &inputSchema, TupleSpec &attsToOperateOn, TupleSpec &additionalAtts,
           std::vector<int> &whereEveryoneGoes, size_t numPartitions) : numPartitions(numPartitions), whereEveryoneGoes(whereEveryoneGoes) {

    // used to manage attributes and set up the output
    TupleSetSetupMachine myMachine(inputSchema);

    // figure out the key att
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    keyAtt = matches[0];

    // now, figure out the attributes that we need to store in the hash table
    useTheseAtts = myMachine.match(additionalAtts);
  }

  ~JoinBuildSink() override {
    if (columns != nullptr)
      delete columns;
  }

  Handle<Object> createNewOutputContainer() override {

    // we simply create a new vector of join maps to store the output
    Handle<Vector<HashedJoinTuple<RHSType>>> returnVal = makeObject<Vector<HashedJoinTuple<RHSType>>>();

    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {

    // get the map we are adding to
    Handle<Vector<HashedJoinTuple<RHSType>>> writeMe = unsafeCast<Vector<HashedJoinTuple<RHSType>>>(writeToMe);

    // make a vector for the RHSType columns and one extra for the hash
    if (columns == nullptr)
      columns = new void *[whereEveryoneGoes.size()];

    // after the hash attribute is the actual join tuple
    int counter = 0;
    for (counter = 0; counter < whereEveryoneGoes.size(); counter++) {
      columns[counter] = (void *) &(input->getColumn<int>(useTheseAtts[whereEveryoneGoes[counter]]));
    }

    // the hash attribute comes first
    std::vector<size_t> &keyColumn = input->getColumn<size_t>(keyAtt);

    // go through each value in the tuple set (key column is used since every column is expected to be the same size)
    for (int i = 0; i < keyColumn.size(); i++) {

      // create an empty element
      try {

        // push an empty element at the back
        writeMe->push_back();

      } catch (NotEnoughSpace &n) {

        // erase the hashes that we processed
        keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);

        // remove the join tuples we have processed.
        truncate<RHSType>(i, 0, columns);

        // forward the exception
        throw n;
      }

      // now try to do the copy
      try {

        // get the record
        HashedJoinTuple<RHSType> &tmp = (*writeMe)[i];

        // copy the hash
        tmp.hash = keyColumn[i];

        // copy the join record
        pack(tmp.tuple, i, 0, columns);

        // if the copy didn't work, pop the value off
      } catch (NotEnoughSpace &n) {

        // pop the back
        writeMe->pop_back();

        // erase the hashes that we processed
        keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);

        // remove the join tuples we have processed.
        truncate<RHSType>(i, 0, columns);

        // forward the exception
        throw n;
      }
    }
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("Join sink can not write out a page."); }

};

}