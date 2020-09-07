#include <PDBClient.h>
#include <GenericWork.h>
#include <random>
#include "TRABlock.h"
#include "sharedLibraries/headers/TensorScanner.h"
#include "sharedLibraries/headers/RMMJoin.h"
#include "sharedLibraries/headers/RMMAggregation.h"
#include "sharedLibraries/headers/RMMDuplicateMultiSelection.h"
#include "sharedLibraries/headers/TensorWriter.h"

using namespace pdb;
using namespace pdb::matrix;

// some constants for the test
const size_t blockSize = 64;
const uint32_t matrixRows = 400;
const uint32_t matrixColumns = 400;
const uint32_t numRows = 40;
const uint32_t numCols = 40;
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
    while(i != tuplesToSend.size()) {

        // use temporary allocation block
        const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

        // put the chunks here
        Handle<Vector<Handle<TRABlock>>> data = pdb::makeObject<Vector<Handle<TRABlock>>>();

        try {

            // put stuff into the vector
            for(; i < tuplesToSend.size();) {

                // allocate a matrix
                Handle<TRABlock> myInt = makeObject<TRABlock>(tuplesToSend[i].first, tuplesToSend[i].second, 0,
                                                              matrixRows / numRows, matrixColumns / numCols, 1);

                // init the values
                float *vals = myInt->data->data->c_ptr();
                for (int v = 0; v < (matrixRows / numRows) * (matrixColumns / numCols); ++v) {
                    vals[v] = 1.0f * v;
                }

                // we add the matrix to the block
                data->push_back(myInt);

                // go to the next one
                ++i;

                if(data->size() == 50) {
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

int main(int argc, char* argv[]) {

    // make a client
    pdb::PDBClient pdbClient(8108, "localhost");

    /// 1. Register the classes

    // now, register a type for user data
    pdbClient.registerType("libraries/libRMMAggregation.so");
    pdbClient.registerType("libraries/libRMMJoin.so");
    pdbClient.registerType("libraries/libRMMDuplicatedMultiSelection.so");
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

    /// 4. Make query graph an run query

    // for allocations
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    Handle <Computation> readA = makeObject <TensorScanner>("myData", "A");
    Handle <Computation> readB = makeObject <TensorScanner>("myData", "B");
    Handle <Computation> duplicateA = makeObject<RMMDuplicateMultiSelection>(2, numRows); //Binhang: since currently it is all square matrix, so it does not matter.
    duplicateA->setInput(readA);
    Handle <Computation> duplicateB = makeObject<RMMDuplicateMultiSelection>(0, numRows);
    duplicateB->setInput(readB);
    Handle <Computation> join = makeObject <RMMJoin>();
    join->setInput(0, duplicateA);
    join->setInput(1, duplicateB);
    Handle<Computation> myAggregation = makeObject<RMMAggregation>();
    myAggregation->setInput(join);
    Handle<Computation> myWriter = makeObject<TensorWriter>("myData", "C");
    myWriter->setInput(myAggregation);


    //Todo:: This requires to shuffle duplicateA and duplicateB both by dim:[0,2]

    // shutdown the server
    pdbClient.shutDownServer();

    return 0;
}