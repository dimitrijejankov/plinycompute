#include <PDBClient.h>
#include <SharedEmployee.h>

int main(int argc, char* argv[]) {

  const size_t blockSize = 64;

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  // now, register a type for user data
  pdbClient.registerType("libraries/libSharedEmployee.so");

  // now, create a new database
  pdbClient.createDatabase("chris_db");

  // now, create a new set in that database
  pdbClient.createSet<SharedEmployee>("chris_db", "chris_set");

  int count = 0;
  for(int j = 0; j < 5; j++) {

    // allocate the thing
    pdb::makeObjectAllocatorBlock(blockSize * 1024l, true);

    // allocate the vector
    pdb::Handle<pdb::Vector<pdb::Handle<SharedEmployee>>> storeMe = pdb::makeObject<pdb::Vector<pdb::Handle<SharedEmployee>>>();

    try {

      for (int i = 0; true; i++) {

        pdb::Handle<SharedEmployee> myData;

        if (i % 100 == 0) {
          myData = pdb::makeObject<SharedEmployee>("Frank", count);
        } else {
          myData = pdb::makeObject<SharedEmployee>("Joe Johnson" + to_string(count), count + 45);
        }

        count++;

        storeMe->push_back(myData);
      }

    } catch (pdb::NotEnoughSpace& n) {

      pdbClient.sendData<SharedEmployee>("chris_db", "chris_set", storeMe);
    }
  }

  // grab the iterator
  auto it = pdbClient.getSetIterator<SharedEmployee>("chris_db", "chris_set");

  int i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // print every 1000th
    if(i % 100 == 0) {
      std::cout << *r->getName() << std::endl;
    }

    // go to the next one
    i++;
  }

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}