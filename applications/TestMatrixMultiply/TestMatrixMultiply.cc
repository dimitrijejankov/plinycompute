#include <boost/program_options.hpp>
#include <PDBClient.h>
#include <GenericWork.h>
#include "sharedLibraries/headers/MatrixBlock.h"
#include "sharedLibraries/headers/MatrixScanner.h"
#include "sharedLibraries/headers/MatrixMultiplyJoin.h"
#include "sharedLibraries/headers/MatrixMultiplyAggregation.h"
#include "sharedLibraries/headers/MatrixWriter.h"

using namespace pdb;
using namespace pdb::matrix;
namespace po = boost::program_options;

void initMatrix(
    pdb::PDBClient &pdbClient,
    const std::string &set,
    const size_t &blockSize,
    const uint32_t &matrixRows,
    const uint32_t &matrixColumns,
    const uint32_t &numRows,
    const uint32_t &numCols) {
  // A matrix consists of numRows x numCols grid of chunks and each
  // chunk consists of the appropriate number of values to make the whole
  // matrix matrixRows x matrixColumns.

  // For each chunk, add the MatrixBlocks. Once the allocation block is full
  // or the all MatrixBlocks have been created, send the data to pdbClient.
  uint32_t numChunks = numRows*numCols;
  uint32_t chunk = 0;
  while(chunk != numChunks) {

    // make the allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<MatrixBlock>>> data = pdb::makeObject<Vector<Handle<MatrixBlock>>>();

    try {
      for(; chunk != numChunks; chunk++) {
        uint32_t r = chunk % numRows;
        uint32_t c = chunk / numRows;

        uint32_t numRowsInChunk = matrixRows    / numRows + (r < matrixRows    % numRows ? 1 : 0);
        uint32_t numColsInChunk = matrixColumns / numCols + (c < matrixColumns % numCols ? 1 : 0);

        // allocate a matrix block
        Handle<MatrixBlock> myInt = makeObject<MatrixBlock>(r, c, numRowsInChunk, numColsInChunk);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for (int v = 0; v < numRowsInChunk*numColsInChunk; ++v) {
          vals[v] = 1.0f * (float) v;
        }

        data->push_back(myInt);
      }
    } catch (pdb::NotEnoughSpace &n) {
    }

    // init the records
    getRecord(data);

    // send the data
    pdbClient.sendData<MatrixBlock>("myData", set, data);
  }
}

void printMatrix(
    pdb::PDBClient& pdbClient,
    std::string const& dbName,
    std::string const& setName) {

  auto it = pdbClient.getSetIterator<MatrixBlock>(dbName, setName);

  while(it->hasNextRecord()) {

        // grab the record
        auto r = it->getNextRecord();

        // write out the values
        float *values = r->data->data->c_ptr();
        for(int i = 0; i < r->data->numRows; ++i) {
            for(int j = 0; j < r->data->numCols; ++j) {
                std::cout << values[i * r->data->numCols + j] << ", ";
            }
            std::cout << "\n";
        }

        std::cout << "\n\n";
    }
}

int main(int argc, char* argv[]) {

  po::options_description desc{"Options"};

  size_t blockSize;
  uint32_t matrixRowsA, matrixRowsB, matrixColumnsB;
  uint32_t numRowsA, numRowsB, numColumnsB;

  // specify the options
  desc.add_options()("help,h", "Help screen");
  desc.add_options()("blockSize", po::value<size_t>(&blockSize)->default_value(1024),
      "Block size for allocation");
  desc.add_options()("matrixRowsA", po::value<uint32_t>(&matrixRowsA)->default_value(10000),
      "Number of rows in matrix A");
  desc.add_options()("matrixRowsB", po::value<uint32_t>(&matrixRowsB)->default_value(10000),
      "Number of columns in matrix A and number of rows in matrix B");
  desc.add_options()("matrixColumnsB", po::value<uint32_t>(&matrixColumnsB)->default_value(10000),
      "Number of columns in matrix B");
  desc.add_options()("numRowsA", po::value<uint32_t>(&numRowsA)->default_value(200),
      "Number of rows in each chunk of matrix A");
  desc.add_options()("numRowsB", po::value<uint32_t>(&numRowsB)->default_value(200),
      "Number of columns in each chunk of matrix A and number of rows in each chunk of matrix B");
  desc.add_options()("numColumnsB", po::value<uint32_t>(&numColumnsB)->default_value(200),
      "Number of columns in each chunk of matrix B");

  // grab the options
  po::variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  // did somebody ask for help?
  if (vm.count("help")) {
    std::cout << desc << '\n';
    return 0;
  }

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libMatrixBlock.so");
  pdbClient.registerType("libraries/libMatrixBlockData.so");
  pdbClient.registerType("libraries/libMatrixBlockMeta.so");
  pdbClient.registerType("libraries/libMatrixMultiplyAggregation.so");
  pdbClient.registerType("libraries/libMatrixMultiplyJoin.so");
  pdbClient.registerType("libraries/libMatrixScanner.so");
  pdbClient.registerType("libraries/libMatrixWriter.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<MatrixBlock>("myData", "A");
  pdbClient.createSet<MatrixBlock>("myData", "B");
  pdbClient.createSet<MatrixBlock>("myData", "C");

  /// 3. Fill in the data (single threaded)

  initMatrix(pdbClient, "A", blockSize, matrixRowsA, matrixRowsB,    numRowsA, numRowsB);
  initMatrix(pdbClient, "B", blockSize, matrixRowsB, matrixColumnsB, numRowsB, numColumnsB);

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  Handle <Computation> readA = makeObject <MatrixScanner>("myData", "A");
  Handle <Computation> readB = makeObject <MatrixScanner>("myData", "B");
  Handle <Computation> join = makeObject <MatrixMultiplyJoin>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle<Computation> myAggregation = makeObject<MatrixMultiplyAggregation>();
  myAggregation->setInput(join);
  Handle<Computation> myWriter = makeObject<MatrixWriter>("myData", "C");
  myWriter->setInput(myAggregation);

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  pdbClient.executeComputations({ myWriter });

  /// 5. Get the set and print
  printMatrix(pdbClient, "myData", "C");

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}
