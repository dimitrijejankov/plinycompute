#include <PDBClient.h>
#include <GenericWork.h>
#include "sharedLibraries/headers/Matrix3DScanner.h"
#include "sharedLibraries/headers/Matrix3DWriter.h"
#include "sharedLibraries/headers/MatrixConv3DJoin.h"

using namespace pdb;
using namespace pdb::matrix_3d;

// some constants for the test
const uint32_t numChannels = 3;
const size_t blockSize = 6;
const uint32_t matrixRows = 4000;
const uint32_t matrixColumns = 4000;
const uint32_t numX = 4;
const uint32_t numY = 4;
const uint32_t numZ = 4;

void initMatrix(pdb::PDBClient &pdbClient, const std::string &set) {

  // fill the vector up
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> tuplesToSend;
  for (uint32_t x = 0; x < numX; x++) {
    for (uint32_t y = 0; y < numY; y++) {
      for (uint32_t z = 0; z < numZ; z++) {
        tuplesToSend.emplace_back(std::make_tuple(x, y, z));
      }
    }
  }

  // make the allocation block
  size_t i = 0;
  while (i != tuplesToSend.size()) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<MatrixBlock3D>>> data = pdb::makeObject<Vector<Handle<MatrixBlock3D>>>();

    try {

      // put stuff into the vector
      for (; i < tuplesToSend.size(); ++i) {

        // allocate a matrix
        Handle<MatrixBlock3D> myInt = makeObject<MatrixBlock3D>(std::get<0>(tuplesToSend[i]),
                                                                std::get<1>(tuplesToSend[i]),
                                                                std::get<2>(tuplesToSend[i]),
                                                                blockSize,
                                                                blockSize,
                                                                blockSize,
                                                                numChannels);

        myInt->data->isTopBorder = std::get<1>(tuplesToSend[i]) == 0;
        myInt->data->isLeftBorder = std::get<0>(tuplesToSend[i]) == 0;
        myInt->data->isFrontBorder = std::get<2>(tuplesToSend[i]) == 0;
        myInt->data->isBottomBorder = std::get<1>(tuplesToSend[i]) == numX - 1;
        myInt->data->isRightBorder = std::get<0>(tuplesToSend[i]) == numY - 1;
        myInt->data->isBackBorder = std::get<2>(tuplesToSend[i]) == numZ - 1;

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for (uint32_t v = 0; v < blockSize * blockSize * blockSize * numChannels; ++v) {
          vals[v] = 1.0f * v;
        }

        // we add the matrix to the block
        data->push_back(myInt);
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(data);

    // send the data a bunch of times
    pdbClient.sendData<MatrixBlock3D>("myData", set, data);

    // log that we stored stuff
    std::cout << "Stored " << data->size() << " !\n";
  }

}

int main(int argc, char *argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libMatrix3DScanner.so");
  pdbClient.registerType("libraries/libMatrix3DWriter.so");
  pdbClient.registerType("libraries/libMatrixBlock3D.so");
  pdbClient.registerType("libraries/libMatrixBlockData3D.so");
  pdbClient.registerType("libraries/libMatrixBlockMeta3D.so");
  pdbClient.registerType("libraries/libMatrixConv3DJoin.so");
  pdbClient.registerType("libraries/libMatrixConvResult.so");


  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<pdb::matrix_3d::MatrixBlock3D>("myData", "A");
  pdbClient.createSet<pdb::matrix_3d::MatrixConvResult>("myData", "B");

  /// 3. Fill in the data (single threaded)

  initMatrix(pdbClient, "A");

  /// 4. Make query graph an run query

  // for allocations
  const pdb::UseTemporaryAllocationBlock tempBlock{128 * 1024 * 1024};

  // create all of the computation objects
  Handle<Computation> a1 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");
  Handle<Computation> a2 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");
  Handle<Computation> a3 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");
  Handle<Computation> a4 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");
  Handle<Computation> a5 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");
  Handle<Computation> a6 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");
  Handle<Computation> a7 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");
  Handle<Computation> a8 = pdb::makeObject<pdb::matrix_3d::Matrix3DScanner>("myData", "A");

  // g
  Handle<Computation> join = pdb::makeObject<pdb::matrix_3d::MatrixConv3DJoin>(blockSize);
  join->setInput(0, a1);
  join->setInput(1, a2);
  join->setInput(2, a3);
  join->setInput(3, a4);
  join->setInput(4, a5);
  join->setInput(5, a6);
  join->setInput(6, a7);
  join->setInput(7, a8);

  Handle<Computation> writeStringIntPair = pdb::makeObject<pdb::matrix_3d::Matrix3DWriter>("myData", "B");
  writeStringIntPair->setInput(0, join);

  std::chrono::steady_clock::time_point planner_begin = std::chrono::steady_clock::now();

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  bool success = pdbClient.executeComputations({writeStringIntPair});

  std::chrono::steady_clock::time_point planner_end = std::chrono::steady_clock::now();
  std::cout << "Run convolution for "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(planner_end - planner_begin).count()
            << "[ns]" << '\n';


  /// 5. Get the set from the

  // grab the iterator
  int count = 0;
  auto it = pdbClient.getSetIterator<MatrixConvResult>("myData", "B");
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    count++;
  }

  std::cout << count << "\n";

  // wait a bit before the shutdown
  sleep(4);

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}