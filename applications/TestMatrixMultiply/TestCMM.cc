#include <PDBClient.h>
#include <GenericWork.h>
#include <random>
#include "TRABlock.h"
#include "sharedLibraries/headers/TensorScanner.h"
#include "sharedLibraries/headers/BCMMJoin.h"
#include "sharedLibraries/headers/BCMMAggregation.h"
#include "sharedLibraries/headers/TensorWriter.h"

using namespace pdb;
using namespace pdb::matrix;

// some constants for the test
size_t blockSize = 128;
uint32_t matrixRows = 400;
uint32_t matrixColumns = 400;
uint32_t numRows = 4;
uint32_t numCols = 4;
const bool doNotPrint = true;

void initMatrix(pdb::PDBClient &pdbClient, const std::string &set) {

  // fill the vector up
  std::vector<std::pair<uint32_t, uint32_t>> tuplesToSend;
  for (uint32_t r = 0; r < numRows; r++) {
    for (uint32_t c = 0; c < numCols; c++) {
      tuplesToSend.emplace_back(std::make_pair(r, c));
    }
  }

  // make the allocation block
  size_t i = 0;
  while (i != tuplesToSend.size()) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<TRABlock>>> data = pdb::makeObject<Vector<Handle<TRABlock>>>();

    try {

      // put stuff into the vector
      for (; i < tuplesToSend.size();) {

        // allocate a matrix
        Handle<TRABlock> myInt = makeObject<TRABlock>(tuplesToSend[i].first, tuplesToSend[i].second, 0,
                                                      matrixRows / numRows, matrixColumns / numCols, 1);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for (int v = 0; v < (matrixRows / numRows) * (matrixColumns / numCols); ++v) {
          vals[v] = 1.0;
        }

        // we add the matrix to the block
        data->push_back(myInt);

        // go to the next one
        ++i;

        if (data->size() == 50) {
          break;
        }
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(data);

    // send the data a bunch of times
    pdbClient.sendData<TRABlock>("myData", set, data, 0);

    // log that we stored stuff
    std::cout << "Stored " << data->size() << " !\n";
  }

}

int main(int argc, char *argv[]) {

  std::cout << "block \n";
  std::cin >> blockSize;
  std::cout << "rows : \n";
  std::cin >> matrixRows;
  std::cout << "cols : \n";
  std::cin >> matrixColumns;
  std::cout << "r split : \n";
  std::cin >> numRows;
  std::cout << "c split : \n";
  std::cin >> numCols;

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libBCMMAggregation.so");
  pdbClient.registerType("libraries/libBCMMJoin.so");
  pdbClient.registerType("libraries/libTensorScanner.so");
  pdbClient.registerType("libraries/libTensorWriter.so");


  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<TRABlock>("myData", "A");
  pdbClient.createSet<TRABlock>("myData", "B");
  pdbClient.createSet<TRABlock>("myData", "C");

  /// 3. Fill in the data (single threaded)

  initMatrix(pdbClient, "A");
  initMatrix(pdbClient, "B");

  // you have to do this to reference a page set in the TRA interface
  pdbClient.createIndex("myData", "A");
  pdbClient.createIndex("myData", "B");

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  Handle <Computation> readA = makeObject <TensorScanner>("myData", "A");
  Handle <Computation> readB = makeObject <TensorScanner>("myData", "B");
  Handle <Computation> join = makeObject <BCMMJoin>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle<Computation> myAggregation = makeObject<BCMMAggregation>();
  myAggregation->setInput(join);
  Handle<Computation> myWriter = makeObject<TensorWriter>("myData", "C");
  myWriter->setInput(myAggregation);

  
std::chrono::steady_clock::time_point shuffle1_begin = std::chrono::steady_clock::now();
pdbClient.shuffle("myData:A", {1}, "AShuffled");
std::chrono::steady_clock::time_point shuffle1_end = std::chrono::steady_clock::now();

std::chrono::steady_clock::time_point shuffle2_begin = std::chrono::steady_clock::now();
pdbClient.shuffle("myData:B", {0}, "BShuffled");
std::chrono::steady_clock::time_point shuffle2_end = std::chrono::steady_clock::now();

std::chrono::steady_clock::time_point join_begin = std::chrono::steady_clock::now();
pdbClient.localJoin("AShuffled", {1}, "BShuffled", {0}, { myWriter }, "ABJoined",
              "OutForJoinedFor_equals_0JoinComp2", "OutFor_joinRec_5JoinComp2");
std::chrono::steady_clock::time_point join_end = std::chrono::steady_clock::now();


std::chrono::steady_clock::time_point shuffle3_begin = std::chrono::steady_clock::now();
pdbClient.shuffle("ABJoined", {0, 1}, "ABJoinedShuffled");
std::chrono::steady_clock::time_point shuffle3_end = std::chrono::steady_clock::now();

std::chrono::steady_clock::time_point agg_begin = std::chrono::steady_clock::now();
pdbClient.localAggregation("ABJoinedShuffled", {0, 1},  "Final");
std::chrono::steady_clock::time_point agg_end = std::chrono::steady_clock::now();

std::chrono::steady_clock::time_point mat_begin = std::chrono::steady_clock::now();
pdbClient.materialize("myData", "C", "Final");
std::chrono::steady_clock::time_point mat_end = std::chrono::steady_clock::now();


std::cout << "Computation run shuffle1 for " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(shuffle1_end - shuffle1_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count() << "[s]" << '\n';
std::cout << "Computation run shuffle2 for " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(shuffle2_end - shuffle2_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count() << "[s]" << '\n';
std::cout << "Computation run join for " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(join_end - join_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count() << "[s]" << '\n';
std::cout << "Computation run shuffle3 for " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(shuffle3_end - shuffle3_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count() << "[s]" << '\n';
std::cout << "Computation run agg for " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(agg_end - agg_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count() << "[s]" << '\n';
std::cout << "Computation run mat for " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(mat_end - mat_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count() << "[s]" << '\n';
std::cout << "Computation run total for " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(mat_end - shuffle1_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count() << "[s]" << '\n';


  // grab the iterator
  auto it = pdbClient.getSetIterator<TRABlock>("myData", "C");
  int32_t count = 0;
  while (it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    //r->print();
    count++;
  }

  std::cout << "Count " << count << '\n';

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}
