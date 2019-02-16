#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>

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

  // init the worker threads of this server
  auto workers = make_shared<pdb::PDBWorkerQueue>(make_shared<pdb::PDBLogger>("worker.log"),  10);

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {
    cnt++;
  });

  atomic_int count;
  count = 0;


  std::vector<std::string> names = {"Frank", "Joe", "Mark", "David", "Zoe"};
  for(int j = 0; j < 5; j++) {

    // the thread
    int thread = j;

    // grab a worker
    pdb::PDBWorkerPtr myWorker = workers->getWorker();

    // start the thread
    pdb::PDBWorkPtr myWork = make_shared<pdb::GenericWork>([&, thread](PDBBuzzerPtr callerBuzzer) {

      // allocate the thing
      pdb::makeObjectAllocatorBlock(blockSize * 1024l * 1024, true);

      // allocate the vector
      pdb::Handle<pdb::Vector<pdb::Handle<SharedEmployee>>> storeMe = pdb::makeObject<pdb::Vector<pdb::Handle<SharedEmployee>>>();

      try {

        for (int i = 0; true; i++) {

          pdb::Handle<SharedEmployee> myData;

          if (i % 100 == 0) {
            myData = pdb::makeObject<SharedEmployee>(names[thread] + " Frank", count);
          } else {
            myData = pdb::makeObject<SharedEmployee>(names[thread] + " " + to_string(count), count + 45);
          }

          count++;

          storeMe->push_back(myData);
        }

      } catch (pdb::NotEnoughSpace &n) {

        pdbClient.sendData<SharedEmployee>("chris_db", "chris_set", storeMe);
      }

      // excellent everything worked just as expected
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    myWorker->execute(myWork, tempBuzzer);
  }

  // wait until all the nodes are finished
  while (counter < 5) {
    tempBuzzer->wait();
  }


  // grab the iterator
  auto it = pdbClient.getSetIterator<SharedEmployee>("chris_db", "chris_set");

  int i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // print every 100th
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