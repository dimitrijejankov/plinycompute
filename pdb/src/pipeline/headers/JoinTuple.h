/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#ifndef JOIN_TUPLE_H
#define JOIN_TUPLE_H

#include "TupleSet.h"

namespace pdb {

template<typename T>
void copyFrom(T &out, Handle<T> &in) {
  out = *in;
}

template<typename T>
void copyFrom(Handle<T> &out, Handle<T> &in) {
  out = in;
}

template<typename T>
void copyTo(T &out, Handle<T> &in) {
  char *location = (char *) &out;
  location -= REF_COUNT_PREAMBLE_SIZE;
  in = (RefCountedObject<T> *) location;
}

template<typename T>
void copyTo(Handle<T> &out, Handle<T> &in) {
  in = out;
}

// this checks to see if the class is abstract
// used like: decltype (IsAbstract <Foo> :: val) a;
// the type of a will be Handle <Foo> if foo is abstract, and Foo otherwise.
//
template<typename T>
struct IsAbstract {
  template<typename U>
  static U test(U x, int);

  template<typename U, typename ...Rest>
  static Handle<U> test(U &x, Rest...);

  static decltype(test<T>(*((T *) 0), 1)) val;
};

// all join tuples decend from this
class JoinTupleBase {};

// this template is used to hold a tuple made of one row from each of a number of columns in a TupleSet
template<typename HoldMe, typename MeTo>
class JoinTuple : public JoinTupleBase {

 public:

  // this stores the base type
  decltype(IsAbstract<HoldMe>::val) myData;

  // and this is the recursion
  MeTo myOtherData;

  static void *allocate(TupleSet &processMe, int where) {
    std::cout << "Creating column for type " << getTypeName<Handle<HoldMe>>() << " at position " << where << "\n";
    auto *me = new std::vector<Handle<HoldMe>>;
    processMe.addColumn(where, me, true);
    return me;
  }

  void copyFrom(void *input, int whichPos) {
    // std :: cout << "Packing column for type " << getTypeName <decltype (IsAbstract <HoldMe> :: val)> () << " at position " << whichPos << "\n";
    std::vector<Handle<HoldMe>> &me = *((std::vector<Handle<HoldMe>> *) input);
    pdb::copyFrom(myData, me[whichPos]);
  }

  void copyTo(void *input, int whichPos) {
    std::vector<Handle<HoldMe>> &me = *((std::vector<Handle<HoldMe>> *) input);

    if (whichPos >= me.size()) {
      Handle<HoldMe> temp;
      pdb::copyTo(myData, temp);
      me.push_back(temp);
    } else {
      pdb::copyTo(myData, me[whichPos]);
    }
  }

  static void truncate(void *input, int i) {
    std::vector<Handle<HoldMe>> &valColumn = *((std::vector<Handle<HoldMe>> *) (input));
    valColumn.erase(valColumn.begin(), valColumn.begin() + i);
  }

  static void eraseEnd(void *input, int i) {
    std::vector<Handle<HoldMe>> &valColumn = *((std::vector<Handle<HoldMe>> *) (input));
    valColumn.resize(i);
  }

};

/***** CODE TO CREATE A SET OF ATTRIBUTES IN A TUPLE SET *****/

// this adds a new column to processMe of type TypeToCreate.  This is added at position offset + positions[whichPos]
template<typename TypeToCreate>
typename std::enable_if<sizeof(TypeToCreate::myOtherData) == 0, void>::type createCols(void **putUsHere,
                                                                                       TupleSet &processMe,
                                                                                       int offset,
                                                                                       int whichPos,
                                                                                       std::vector<int> positions) {
  putUsHere[whichPos] = TypeToCreate::allocate(processMe, offset + positions[whichPos]);
}

// recursive version of the above
template<typename TypeToCreate>
typename std::enable_if<sizeof(TypeToCreate::myOtherData) != 0, void>::type createCols(void **putUsHere,
                                                                                       TupleSet &processMe,
                                                                                       int offset,
                                                                                       int whichPos,
                                                                                       std::vector<int> positions) {
  putUsHere[whichPos] = TypeToCreate::allocate(processMe, offset + positions[whichPos]);
  createCols<decltype(TypeToCreate::myOtherData)>(putUsHere, processMe, offset, whichPos + 1, positions);
}

/***** CODE TO PACK A JOIN TUPLE FROM A SET OF VALUES SPREAD ACCROSS COLUMNS *****/

// this is the non-recursive version of pack; called if the type does NOT have a field called "myData", in which case
// we can just directly copy the data
template<typename TypeToPack>
typename std::enable_if<sizeof(TypeToPack::myOtherData) == 0, void>::type pack(TypeToPack &arg,
                                                                               int whichPosInTupleSet,
                                                                               int whichVec,
                                                                               void **us) {
  arg.copyFrom(us[whichVec], whichPosInTupleSet);
}

// this is the recursive version of pack; called if the type has a field called "myData" to which we can recursively
// pack values to.  Basically, what it does is to accept a pointer to a list of pointers to various std :: vector 
// objects.  We are going to recurse through the list of vectors, and for each vector, we record the entry at 
// the position whichPosInTupleSet
template<typename TypeToPack>
typename std::enable_if<sizeof(TypeToPack::myOtherData) != 0, void>::type pack(TypeToPack &arg,
                                                                               int whichPosInTupleSet,
                                                                               int whichVec,
                                                                               void **us) {

  arg.copyFrom(us[whichVec], whichPosInTupleSet);
  pack(arg.myOtherData, whichPosInTupleSet, whichVec + 1, us);
}

/***** CODE TO UNPACK A JOIN TUPLE FROM A SET OF VALUES SPREAD ACCROSS COLUMNS *****/

// this is the non-recursive version of unpack
template<typename TypeToUnPack>
typename std::enable_if<sizeof(TypeToUnPack::myOtherData) == 0, void>::type unpack(TypeToUnPack &arg,
                                                                                   int whichPosInTupleSet,
                                                                                   int whichVec,
                                                                                   void **us) {
  arg.copyTo(us[whichVec], whichPosInTupleSet);
}

// this is analagous to pack, except that it unpacks this tuple into an array of vectors
template<typename TypeToUnPack>
typename std::enable_if<sizeof(TypeToUnPack::myOtherData) != 0, void>::type unpack(TypeToUnPack &arg,
                                                                                   int whichPosInTupleSet,
                                                                                   int whichVec,
                                                                                   void **us) {

  arg.copyTo(us[whichVec], whichPosInTupleSet);
  unpack(arg.myOtherData, whichPosInTupleSet, whichVec + 1, us);
}

/***** CODE TO ERASE DATA FROM THE END OF A SET OF VECTORS *****/

// this is the non-recursive version of eraseEnd
template<typename TypeToTruncate>
typename std::enable_if<sizeof(TypeToTruncate::myOtherData) == 0, void>::type eraseEnd(int i, int whichVec, void **us) {

  TypeToTruncate::eraseEnd(us[whichVec], i);
}

// recursive version
template<typename TypeToTruncate>
typename std::enable_if<sizeof(TypeToTruncate::myOtherData) != 0, void>::type eraseEnd(int i, int whichVec, void **us) {

  TypeToTruncate::eraseEnd(us[whichVec], i);
  eraseEnd<decltype(TypeToTruncate::myOtherData)>(i, whichVec + 1, us);
}

/***** CODE TO TRUNCATE A SET OF VECTORS *****/

// this is the non-recursive version of truncate
template<typename TypeToTruncate>
typename std::enable_if<sizeof(TypeToTruncate::myOtherData) == 0, void>::type truncate(int i, int whichVec, void **us) {

  TypeToTruncate::truncate(us[whichVec], i);
}

// this function goes through a list of vectors, and truncates each of them so that the first i entries of each vector is removed
template<typename TypeToTruncate>
typename std::enable_if<sizeof(TypeToTruncate::myOtherData) != 0, void>::type truncate(int i, int whichVec, void **us) {

  TypeToTruncate::truncate(us[whichVec], i);
  truncate<decltype(TypeToTruncate::myOtherData)>(i, whichVec + 1, us);
}

}

#endif
