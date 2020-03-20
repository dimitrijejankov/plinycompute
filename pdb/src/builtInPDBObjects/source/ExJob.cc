#include <ExJob.h>
#include <PDBPhysicalAlgorithm.h>

/**
 * Returns all the sets that are going to be materialized after the job is executed
 * @return - a vector of pairs the frist value is the database name, the second value is the set name
 */
std::vector<std::pair<std::string, std::string>> pdb::ExJob::getSetsToMaterialize() {

  // get the sets to materialize
  const auto& sets = physicalAlgorithm->getSetsToMaterialize();

  // allocate the output container
  std::vector<std::pair<std::string, std::string>> out = { {"myData", "B"} };

  // return it
  return std::move(out);
}

/**
 * Returns the actual sets we are scanning, it assumes that we are doing that. Check that with @see isScanningSet
 * @return get the scanning set
 */
vector<pair<string, string>> pdb::ExJob::getScanningSets() {

  // return the scanning set
  return std::move(physicalAlgorithm->getSetsToScan());
}

/**
 * True if, the source is an actual set and not an intermediate set
 * @return true if it is, false otherwise
 */
bool pdb::ExJob::isScanningSet() {
  return !physicalAlgorithm->getSetsToScan().empty();
}

/**
 * Returns the type of the output container, that the materializing sets are going to have
 * @return the type
 */
pdb::PDBCatalogSetContainerType pdb::ExJob::getOutputSetContainer() {
  return physicalAlgorithm->getOutputContainerType();
}
