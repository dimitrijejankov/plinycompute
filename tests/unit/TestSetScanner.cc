#include <gtest/gtest.h>

#include <SetScanner.h>
#include <Employee.h>
#include <PDBBufferManagerImpl.h>
#include <PDBAbstractPageSet.h>
#include <gmock/gmock-generated-function-mockers.h>
#include <gmock/gmock-more-actions.h>

namespace pdb {

class MockPageSet : public pdb::PDBAbstractPageSet {
public:

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD0(getNumPages, size_t ());

  MOCK_METHOD0(resetPageSet, void ());
};

TEST(SetScannerTest, Test1) {

  // create the buffer manager
  pdb::PDBBufferManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64 * 1024, 16, "metadata", ".");

  // init 5 pages
  for (int j = 0; j < 5; ++j) {

    // get page
    auto page = myMgr.getPage(make_shared<pdb::PDBSet>("db", "set"), j);

    // set the allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{page->getBytes(), 64 * 1024};

    // allocate the vector
    pdb::Handle<pdb::Vector<pdb::Handle<pdb::Employee>>> storeMe = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Employee>>>();

    try {

      // fill it up
      for (int i = 0; true; i++) {

        pdb::Handle<pdb::Employee> myData;

        if (i % 3 == 0) {
          myData = pdb::makeObject<pdb::Employee>("Ann Frank", i);
        } else {
          myData = pdb::makeObject<pdb::Employee>("Tom Frank", i + 45);
        }

        storeMe->push_back(myData);
      }
    } catch (pdb::NotEnoughSpace &n) {
      getRecord (storeMe);
    }
  }

  // init the scanner
  pdb::SetScanner<pdb::Employee> scanner("db", "set");

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSet> pageSet = std::make_shared<MockPageSet>();

  uint64_t counter = 0;

  // make sure the mock function returns true
  ON_CALL(*pageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        if(counter >= 5) {
          return (PDBPageHandle) nullptr;
        }

        return myMgr.getPage(make_shared<pdb::PDBSet>("db", "set"), counter++);
      }
  ));

  // it should call send object exactly six times
  EXPECT_CALL(*pageSet, getNextPage(testing::An<size_t>())).Times(6);

  // page set ptr
  auto ptr = std::dynamic_pointer_cast<pdb::PDBAbstractPageSet>(pageSet);

  // get the compute source
  std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                        std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("", "", "", 0, PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER))}};
  auto dataSource = scanner.getComputeSource(ptr, 15, 0, params);

  // and here is the chunk
  TupleSetPtr curChunk;

  while ((curChunk = dataSource->getNextTupleSet()) != nullptr) {

    // check the number of columns
    EXPECT_EQ(curChunk->getNumColumns(), 1);

    auto emps = curChunk->getColumn<Handle<pdb::Employee>>(0);

    for(auto &e : emps) {

      EXPECT_TRUE(*e->getName() == "Ann Frank" || *e->getName() == "Tom Frank");
      EXPECT_TRUE((e->getAge() >= 0 && e->getAge() < 3) || (e->getAge() >= 45 || e->getAge() < 48));
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}
