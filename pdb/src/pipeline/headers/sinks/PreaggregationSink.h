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

#ifndef PDB_PREAGGREGATIONSINK_H
#define PDB_PREAGGREGATIONSINK_H

namespace pdb {

/*
 * This is a forward declaration for the implementation of method PreaggregationSink::writeOut.
 * We need to write the implementation for this method as a separate function because we need
 * to write different template specializations for the default VTAdder and Converter types.
 * In C++ you're generally not allowed to specialize just a single method; you would need to
 * specialize the entire class. See here: https://stackoverflow.com/a/16779361
 */
template<class KeyType, class TempValueType, class ValueType, class VTAdder, class Converter>
class WriteOutFunctor;

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
  Converter converter; // Encapsulates function to convert ValueType to TempValueType

  WriteOutFunctor<KeyType, TempValueType, ValueType, VTAdder, Converter> writeOutFunctor;

 public:

  friend class WriteOutFunctor<KeyType, TempValueType, ValueType, VTAdder, Converter>;

  PreaggregationSink(TupleSpec &inputSchema, TupleSpec &attsToOperateOn, size_t numPartitions) :
      numPartitions(numPartitions),
      vtadder(),
      converter(),
      writeOutFunctor() {
    // TODO: if adder/converter/functor become PDB Objects, need to instantiate them here
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
    writeOutFunctor(this, input, writeToMe);


  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("PreaggregationSink can not write out a page."); }

};

// This is the definition for writeOut with both a user-defined adder and user-defined converter.
template<class KeyType, class TempValueType, class ValueType, class VTAdder, class Converter>
class WriteOutFunctor {
 public:
  void operator() (
      PreaggregationSink<KeyType, TempValueType, ValueType, VTAdder, Converter> *thiss,
      TupleSetPtr input,
      Handle<Object> &writeToMe) {
    // cast the thing to the map of maps
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> vectorOfMaps = unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(thiss->whichAttToHash);
    std::vector<TempValueType> &tempValueColumn = input->getColumn<TempValueType>(thiss->whichAttToAggregate);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // hash the key
      auto hash = hashHim(keyColumn[i]);

      // get the map we are adding to
      Map<KeyType, ValueType> &myMap = (*(*vectorOfMaps)[hash % thiss->numPartitions]);

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
          thiss->converter.convert(tempValueColumn[i], temp);

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
          temp = thiss->vtadder.add(copy, tempValueColumn[i]);

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
};

/*
 * This is the specialization for the default adder, with a user-defined converter.
 */
template<class KeyType, class TempValueType, class ValueType, class Converter>
class WriteOutFunctor<KeyType, TempValueType, ValueType, VTDefault, Converter> {
 public:
  void operator() (
      PreaggregationSink<KeyType, TempValueType, ValueType, VTDefault, Converter> *thiss,
      TupleSetPtr input,
      Handle<Object> &writeToMe) {
    // cast the thing to the map of maps
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> vectorOfMaps = unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(thiss->whichAttToHash);
    std::vector<TempValueType> &tempValueColumn = input->getColumn<TempValueType>(thiss->whichAttToAggregate);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // hash the key
      auto hash = hashHim(keyColumn[i]);

      // get the map we are adding to
      Map<KeyType, ValueType> &myMap = (*(*vectorOfMaps)[hash % thiss->numPartitions]);

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
          // NOTE: This is the first of only 2 lines that are changed.
          thiss->converter.convert(tempValueColumn[i], temp);

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
          // NOTE: This is the second of only 2 lines that are changed.
          temp = copy + tempValueColumn[i];

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
};


// This is the specialization for the default converter, with a user-defined adder.
template<class KeyType, class TempValueType, class ValueType, class VTAdder>
class WriteOutFunctor<KeyType, TempValueType, ValueType, VTAdder, ConvertDefault> {
 public:
  void operator() (
      PreaggregationSink<KeyType, TempValueType, ValueType, VTAdder, ConvertDefault> *thiss,
      TupleSetPtr input,
      Handle<Object> &writeToMe) {
    // cast the thing to the map of maps
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> vectorOfMaps = unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(thiss->whichAttToHash);
    std::vector<TempValueType> &tempValueColumn = input->getColumn<TempValueType>(thiss->whichAttToAggregate);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // hash the key
      auto hash = hashHim(keyColumn[i]);

      // get the map we are adding to
      Map<KeyType, ValueType> &myMap = (*(*vectorOfMaps)[hash % thiss->numPartitions]);

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
          // Note: this will implicitly call ValueType's overloaded assignment operator to convert the
          // TempValueType to a ValueType.
          // This is the first of only 2 lines that are changed.
          *temp = tempValueColumn[i];

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
          // NOTE: This is the second of only 2 lines that are changed.
          temp = thiss->vtadder.add(copy, tempValueColumn[i]);

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
};


// This is the specialization for both the default adder and default converter.
template<class KeyType, class TempValueType, class ValueType>
class WriteOutFunctor<KeyType, TempValueType, ValueType, VTDefault, ConvertDefault> {
 public:
  void operator() (
      PreaggregationSink<KeyType, TempValueType, ValueType, VTDefault, ConvertDefault> *thiss,
      TupleSetPtr input,
      Handle<Object> &writeToMe) {
    // cast the thing to the map of maps
    Handle<Vector<Handle<Map<KeyType, ValueType>>>> vectorOfMaps = unsafeCast<Vector<Handle<Map<KeyType, ValueType>>>>(writeToMe);

    // get the input columns
    std::vector<KeyType> &keyColumn = input->getColumn<KeyType>(thiss->whichAttToHash);
    std::vector<TempValueType> &tempValueColumn = input->getColumn<TempValueType>(thiss->whichAttToAggregate);

    // and aggregate everyone
    size_t length = keyColumn.size();
    for (size_t i = 0; i < length; i++) {

      // hash the key
      auto hash = hashHim(keyColumn[i]);

      // get the map we are adding to
      Map<KeyType, ValueType> &myMap = (*(*vectorOfMaps)[hash % thiss->numPartitions]);

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
          // Note: this will implicitly call ValueType's overloaded assignment operator to convert the
          // TempValueType to a ValueType.
          // This is the first of only 2 lines that are changed.
          *temp = tempValueColumn[i];

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
          // NOTE: This is the second of only 2 lines that are changed.
          temp = copy + tempValueColumn[i];

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
};

}

#endif //PDB_PREAGGREGATIONSINK_H
