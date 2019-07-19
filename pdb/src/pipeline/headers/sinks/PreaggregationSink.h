//
// Created by dimitrije on 3/26/19.
//
#include "EqualsLambda.h"
#include "ComputeSink.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include "AbstractAdder.h"
#include "AbstractConverter.h"
#include <vector>
#include <type_traits>

#ifndef PDB_PREAGGREGATIONSINK_H
#define PDB_PREAGGREGATIONSINK_H

namespace pdb {

// runs hashes all of the tuples
template<class KeyType, class TempValueType, class ValueType, class VTAdder, class Converter>
class PreaggregationSink : public ComputeSink {

 private:

  // the attributes to operate on
  int whichAttToHash;
  int whichAttToAggregate;

  // how many partitions do we have
  size_t numPartitions;

  VTAdder vtadder; // Encapsulates function to add ValueType and TempValueType
  Converter converter; // Encapsulates function to convert TempValueType to ValueType


  // Here we define the default and specialized add/convert methods.
  // In the following 4 method definitions, we are using tag dispatching to decide
  // at compile-time whether to use the default or specialized versions.
  // Tag dispatching is a technique that adds an extra dummy argument to a function.
  // As long as the type of the dummy argument can be determined at compile-time, it
  // can be used to select which function overload will be used.
  // For the add method here, the dummy argument should be 'std::is_same<VTAdder, VTDefault>()'.
  // For convert, it should be 'std::is_same<Converter, ConvertDefault>()'.
  //
  // The idea to use tag dispatching came from these stackoverflow posts:
  // https://stackoverflow.com/questions/15598939/how-do-i-use-stdis-integral-to-select-an-implementation
  // https://stackoverflow.com/a/6627748
  //
  // For more info on how tag dispatching works, see here:
  // https://crazycpp.wordpress.com/2014/12/15/tutorial-on-tag-dispatching/

  ValueType add(std::true_type, ValueType& v, TempValueType& t) {
    return v + t;
  }
  // TODO: ValueType can potentially have a large memory footprint. Would it be worthwhile to
  //  do a move in the add methods in order to minimize copies?
  ValueType add(std::false_type, ValueType& v, TempValueType& t) {
    return vtadder.add(v, t);
  }

  void convert(std::true_type, TempValueType& in, ValueType* out) {
    *out = in;
  }

  void convert(std::false_type, TempValueType& in, ValueType* out) {
    converter.convert(in, out);
  }


 public:

  PreaggregationSink(TupleSpec &inputSchema, TupleSpec &attsToOperateOn, size_t numPartitions) :
      numPartitions(numPartitions),
      vtadder(),
      converter() {
    // TODO: if adder/converter become PDB Objects, need to instantiate them here
    // to setup the output tuple set
    TupleSpec empty{};
    TupleSetSetupMachine myMachine(inputSchema, empty);

    // this is the input attribute that we will process
    std::vector<int> matches = myMachine.match(attsToOperateOn);
    whichAttToHash = matches[0];
    whichAttToAggregate = matches[1];
  }

  ~PreaggregationSink() override = default;

  Handle<Object> createNewOutputContainer() override {

    // we simply create a new vector of maps to store the stuff
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> returnVal = makeObject<Vector<Handle<Map<KeyType, ValueType>>>>();

    // create the maps
    for(auto i = 0; i < numPartitions; ++i) {

      // add the map
      returnVal->push_back(makeObject<Map<KeyType, ValueType>>());
    }

    // return the output container
    return returnVal;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {
    // cast the thing to the map of maps
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> vectorOfMaps = unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(whichAttToHash);
    std::vector<TempValueType> &tempValueColumn = input->getColumn<TempValueType>(whichAttToAggregate);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // hash the key
      auto hash = hashHim(keyColumn[i]);

      // get the map we are adding to
      Map<KeyType, ValueType> &myMap = (*(*vectorOfMaps)[hash % numPartitions]);

      // if this key is not already there...
      if (myMap.count(keyColumn[i]) == 0) {

        // this point will record where the value is located
        ValueType *temp = nullptr;

        // try to add the key... this will cause an allocation for a new key/val pair
        try {
          // get the location that we need to write to...
          temp = &(myMap[keyColumn[i]]);

          // if we get an exception, then we could not fit a new key/value pair
        } catch (NotEnoughSpace &n) {

          // if we got here, then we ran out of space, and so we need to delete the already-processed
          // data so that we can try again...
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          tempValueColumn.erase(tempValueColumn.begin(), tempValueColumn.begin() + i);
          throw n;
        }

        // we were able to fit a new key/value pair, so copy over the value
        try {
          // Note: this is the first of only 2 lines that are changed.
          convert(std::is_same<Converter, ConvertDefault>(), tempValueColumn[i], temp);

          // if we could not fit the value...
        } catch (NotEnoughSpace &n) {

          // then we need to erase the key from the map
          myMap.setUnused(keyColumn[i]);

          // and erase all of these guys from the tuple set since they were processed
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          tempValueColumn.erase(tempValueColumn.begin(), tempValueColumn.begin() + i);
          throw n;
        }

        // the key is there
      } else {

        // get the value and a copy of it
        // TODO: is there a way to do this without doing an extra copy?
        ValueType &temp = myMap[keyColumn[i]];
        ValueType copy = temp;

        // and add to the old value, producing a new one
        try {
          // Note: this is the second of only 2 lines that are changed.
          temp = add(std::is_same<VTAdder, VTDefault>(), copy, tempValueColumn[i]);

          // if we got here, then it means that we ram out of RAM when we were trying
          // to put the new value into the hash table
        } catch (NotEnoughSpace &n) {

          // restore the old value
          temp = copy;

          // and erase all of the guys who were processed
          keyColumn.erase(keyColumn.begin(), keyColumn.begin() + i);
          tempValueColumn.erase(tempValueColumn.begin(), tempValueColumn.begin() + i);
          throw n;
        }
      }
    }
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("PreaggregationSink can not write out a page."); }

};

}

#endif //PDB_PREAGGREGATIONSINK_H
