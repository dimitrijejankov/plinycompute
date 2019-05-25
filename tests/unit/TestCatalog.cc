#include "PDBCatalog.h"

#include <iostream>
#include <gtest/gtest.h>
#include <boost/filesystem/operations.hpp>

TEST(CatalogTest, FullTest) {
  // remove the catalog if it exists from a previous run
  boost::filesystem::remove("out.sqlite");

  // create a catalog
  pdb::PDBCatalog catalog("out.sqlite");

  std::string error;

  // create the databases
  EXPECT_TRUE(catalog.registerDatabase(std::make_shared<pdb::PDBCatalogDatabase>("db1"), error));
  EXPECT_TRUE(catalog.registerDatabase(std::make_shared<pdb::PDBCatalogDatabase>("db2"), error));

  // store the number of registred types
  auto numBefore = catalog.numRegisteredTypes();

  // create a type
  EXPECT_TRUE(catalog.registerType(std::make_shared<pdb::PDBCatalogType>(8341, "built-in", "Type1", std::vector<char>()), error));
  EXPECT_TRUE(catalog.registerType(std::make_shared<pdb::PDBCatalogType>(8342, "built-in", "Type2", std::vector<char>()), error));

  // check if we added two new types
  EXPECT_EQ(catalog.numRegisteredTypes() - numBefore, 2);

  // create the set
  EXPECT_TRUE(catalog.registerSet(std::make_shared<pdb::PDBCatalogSet>("db1", "set1", "Type1", 0, pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_NO_CONTAINER), error));
  EXPECT_TRUE(catalog.registerSet(std::make_shared<pdb::PDBCatalogSet>("db1", "set2", "Type1", 0, pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_NO_CONTAINER), error));
  EXPECT_TRUE(catalog.registerSet(std::make_shared<pdb::PDBCatalogSet>("db2", "set3", "Type2", 0, pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_NO_CONTAINER), error));

  // update the set
  EXPECT_TRUE(catalog.incrementSetSize("db1", "set1", 1024, error));
  EXPECT_TRUE(catalog.incrementSetSize("db1", "set1", 2048, error));
  EXPECT_TRUE(catalog.incrementSetSize("db1", "set2", 512, error));
  EXPECT_TRUE(catalog.incrementSetSize("db1", "set2", 1024, error));

  EXPECT_TRUE(catalog.updateSetContainer("db1", "set1", pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER, error));
  EXPECT_TRUE(catalog.updateSetContainer("db1", "set2", pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_MAP_CONTAINER, error));

  // create the nodes
  EXPECT_TRUE(catalog.registerNode(std::make_shared<pdb::PDBCatalogNode>("localhost:8080", "localhost", 8080, "master", 8, 1024, true), error));
  EXPECT_TRUE(catalog.registerNode(std::make_shared<pdb::PDBCatalogNode>("localhost:8081", "localhost", 8081, "worker", 8, 1024, true), error));
  EXPECT_TRUE(catalog.registerNode(std::make_shared<pdb::PDBCatalogNode>("localhost:8082", "localhost", 8082, "worker", 8, 1024, true), error));

  // check the exists functions for databases
  EXPECT_TRUE(catalog.databaseExists("db1"));
  EXPECT_TRUE(catalog.databaseExists("db2"));

  // check the exists functions for sets
  EXPECT_TRUE(catalog.setExists("db1", "set1"));
  EXPECT_TRUE(catalog.setExists("db1", "set2"));
  EXPECT_TRUE(catalog.setExists("db2", "set3"));

  EXPECT_FALSE(catalog.setExists("db1", "set3"));
  EXPECT_FALSE(catalog.setExists("db2", "set1"));
  EXPECT_FALSE(catalog.setExists("db2", "set2"));

  // check if the types exist
  EXPECT_TRUE(catalog.typeExists("Type1"));
  EXPECT_TRUE(catalog.typeExists("Type2"));

  // check if the node exist
  EXPECT_TRUE(catalog.nodeExists("localhost:8080"));
  EXPECT_TRUE(catalog.nodeExists("localhost:8081"));
  EXPECT_TRUE(catalog.nodeExists("localhost:8082"));

  // check the get method for the database
  auto db = catalog.getDatabase("db1");

  EXPECT_EQ(db->name, "db1");
  EXPECT_TRUE(db->createdOn > 0);

  db = catalog.getDatabase("db2");

  EXPECT_EQ(db->name, "db2");
  EXPECT_TRUE(db->createdOn > 0);

  // check the get method for the sets
  auto set = catalog.getSet("db1", "set1");

  EXPECT_EQ(set->name, "set1");
  EXPECT_EQ(set->setIdentifier, "db1:set1");
  EXPECT_EQ(set->database, "db1");
  EXPECT_EQ(*set->type, "Type1");
  EXPECT_EQ(set->containerType, pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER);
  EXPECT_EQ(set->setSize, 1024 + 2048);

  set = catalog.getSet("db1", "set2");

  EXPECT_EQ(set->name, "set2");
  EXPECT_EQ(set->setIdentifier, "db1:set2");
  EXPECT_EQ(set->database, "db1");
  EXPECT_EQ(*set->type, "Type1");
  EXPECT_EQ(set->containerType, pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_MAP_CONTAINER);
  EXPECT_EQ(set->setSize, 1024 + 512);

  set = catalog.getSet("db2", "set3");

  EXPECT_EQ(set->name, "set3");
  EXPECT_EQ(set->setIdentifier, "db2:set3");
  EXPECT_EQ(set->database, "db2");
  EXPECT_EQ(*set->type, "Type2");
  EXPECT_EQ(set->containerType, pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_NO_CONTAINER);
  EXPECT_EQ(set->setSize, 0);

  set = catalog.getSet("db1", "set3");
  EXPECT_TRUE(set == nullptr);

  // check the nodes
  auto node = catalog.getNode("localhost:8080");

  EXPECT_EQ(node->nodeID, "localhost:8080");
  EXPECT_EQ(node->nodeType, "master");
  EXPECT_EQ(node->address, "localhost");
  EXPECT_EQ(node->port, 8080);

  node = catalog.getNode("localhost:8081");

  EXPECT_EQ(node->nodeID, "localhost:8081");
  EXPECT_EQ(node->nodeType, "worker");
  EXPECT_EQ(node->address, "localhost");
  EXPECT_EQ(node->port, 8081);

  node = catalog.getNode("localhost:8082");

  EXPECT_EQ(node->nodeID, "localhost:8082");
  EXPECT_EQ(node->nodeType, "worker");
  EXPECT_EQ(node->address, "localhost");
  EXPECT_EQ(node->port, 8082);

  auto t1 = catalog.getTypeWithoutLibrary(8341);
  auto t2 = catalog.getTypeWithoutLibrary("Type1");

  EXPECT_EQ(t1->name, t2->name);
  EXPECT_EQ(t1->id, t2->id);
  EXPECT_EQ(t1->typeCategory, t2->typeCategory);

  // print out the catalogT
  std::cout << catalog.listNodesInCluster() << std::endl;
  std::cout << catalog.listRegisteredDatabases() << std::endl;
  std::cout << catalog.listUserDefinedTypes() << std::endl;

  // remove the database
  catalog.removeDatabase("db1", error);

  // check the get method for the sets
  EXPECT_TRUE(!catalog.setExists("db1", "set1"));
  EXPECT_TRUE(!catalog.setExists("db1", "set2"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}