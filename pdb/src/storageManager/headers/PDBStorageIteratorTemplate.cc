//
// Created by dimitrije on 2/13/19.
//

namespace pdb {

template<class T>
bool PDBStorageIterator<T>::hasNextRecord() {

  // are we starting out
  if(buffer == nullptr) {

    /// in this case we grab the next page
  }

  // grab the vector
  Handle<Vector<Handle<T>>> pageVector = (Record<Vector<Handle<T>>>*) (buffer.get())->getRootObject();

  // does this page have more records
  if(currRecord < pageVector->size()) {

    /// in this case we grab a new page again
  }

  return false;
}

template<class T>
Handle<T> PDBStorageIterator<T>::getNextRecord() {

  // are we starting out
  if(buffer == nullptr) {

    /// in this case we grab the next page
  }

  // ok we have a buffer mount it
  //const UseTemporaryAllocationBlock tempBlock{ buffer.get(), bufferSize };

  return Handle<T>();
}

}