//
// Created by dimitrije on 1/21/19.
//
#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <random>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <PDBStorageManagerFrontend.h>
#include <GenericWork.h>

namespace pdb {

/**
 * This is the mock communicator we provide to the request handlers
 */
    class CommunicatorMock {

    public:

        MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg));

        MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::SimpleRequestResult>& res, std::string& errMsg));

        MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::BufPinPageResult>& res, std::string& errMsg));

        MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::BufFreezeRequestResult>& res, std::string& errMsg));

    };

    class RequestsMock {
    public:
        //MOCK_METHOD//
    };


    auto getRandomIndices(int numRequestsPerPage, int numPages) {

        // generate the page indices
        std::vector<uint64_t> pageIndices;
        for(int i = 0; i < numRequestsPerPage; ++i) {
            for(int j = 0; j < numPages; ++j) {
                pageIndices.emplace_back(j);
            }
        }

        // shuffle the page indices
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle (pageIndices.begin(), pageIndices.end(), std::default_random_engine(seed));

        return std::move(pageIndices);
    }

// this tests just regular pages

    TEST(StorageManagerFrontendTest, Test1) {
        auto comm = std::make_shared<CommunicatorMock>();
        auto requests = std::make_shared<RequestsMock>();

    }


    int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }

}