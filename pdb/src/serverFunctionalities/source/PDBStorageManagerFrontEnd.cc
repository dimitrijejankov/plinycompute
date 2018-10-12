#include <PDBStorageManagerFrontEnd.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace fs = boost::filesystem;

void pdb::PDBStorageManagerFrontEnd::init() {

  // create the root directory
  fs::path dataPath(getConfiguration()->rootDirectory);
  dataPath.append("/data");

  // create the data directory
  if(!fs::exists(dataPath) && !fs::create_directories(dataPath)) {
    std::cout << "Failed to create the data directory!\n";
  }

  // grab the memory size and he page size
  auto memorySize = getConfiguration()->sharedMemSize;
  auto pageSize = getConfiguration()->pageSize;

  // just a quick sanity check
  if(pageSize == 0 || memorySize == 0) {
    throw std::runtime_error("The memory size or the page size can not be 0");
  }

  // figure out the number of pages we have available
  auto numPages = memorySize / pageSize;

  // finally init the storage manager
  storageManager = std::make_shared<pdb::PDBStorageManager>();
  storageManager->initialize((dataPath / "tempFile___.tmp").string(), pageSize, numPages, (dataPath / "metadata").string(), dataPath.string());
}

pdb::PDBPageHandle pdb::PDBStorageManagerFrontEnd::getPage(pdb::PDBSetPtr whichSet, uint64_t i) {
  return storageManager->getPage(whichSet, i);
}

pdb::PDBPageHandle pdb::PDBStorageManagerFrontEnd::getPage() {
  return storageManager->getPage();
}

pdb::PDBPageHandle pdb::PDBStorageManagerFrontEnd::getPage(size_t minBytes) {
  return storageManager->getPage(minBytes);
}

size_t pdb::PDBStorageManagerFrontEnd::getPageSize() {
  return storageManager->getPageSize();
}
void pdb::PDBStorageManagerFrontEnd::registerHandlers(pdb::PDBServer &forMe) {

}
