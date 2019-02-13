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

  std::cout << pdbClient.getErrorMessage() << std::endl;

  // now, create a new set in that database
  pdbClient.createSet<SharedEmployee>("chris_db", "chris_set");

  pdb::makeObjectAllocatorBlock(blockSize * 1024l * 1024l, true);
  pdb::Handle<pdb::Vector<pdb::Handle<SharedEmployee>>> storeMe = pdb::makeObject<pdb::Vector<pdb::Handle<SharedEmployee>>>();

  for(int j = 0; j < 5; j++) {
    try {

      for (int i = 0; true; i++) {

        pdb::Handle<SharedEmployee> myData;

        if (i % 100 == 0) {
          myData = pdb::makeObject<SharedEmployee>("Frank", i);
        } else {
          myData = pdb::makeObject<SharedEmployee>("Joe Johnson" + to_string(i), i + 45);
        }

        storeMe->push_back(myData);
      }

    } catch (pdb::NotEnoughSpace& n) {

      pdbClient.sendData<SharedEmployee>(std::pair<std::string, std::string>("chris_set", "chris_db"), storeMe);
    }
  }

  sleep(20);

  // shutdown the server
  std::string err;
  pdbClient.shutDownServer(err);

  return 0;
}