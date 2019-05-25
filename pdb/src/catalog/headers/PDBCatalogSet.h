//
// Created by dimitrije on 9/4/18.
//

#ifndef PDB_PDBCATALOGSET_H
#define PDB_PDBCATALOGSET_H

#include <string>
#include <sqlite_orm.h>
#include "PDBCatalogDatabase.h"
#include "PDBCatalogType.h"
#include "PDBCatalogNode.h"

namespace pdb {

/**
 * This is just a definition for the shared pointer on the type
 */
class PDBCatalogSet;
typedef std::shared_ptr<PDBCatalogSet> PDBCatalogSetPtr;

/**
 * The type of the container the pages of this set have
 */
enum PDBCatalogSetContainerType {

  // this means that this set does not have any
  PDB_CATALOG_SET_NO_CONTAINER,

  // this means that the root object is a pdb::Vector
  PDB_CATALOG_SET_VECTOR_CONTAINER,

  // this means that the root object is pdb::Map
  PDB_CATALOG_SET_MAP_CONTAINER
};

/**
 * A class to map the sets
 */
class PDBCatalogSet {
public:

  /**
   * The default constructor for the set required by the orm
   */
  PDBCatalogSet() = default;

  /**
   * The initialization constructor
   * @param name - the name of the set
   * @param database - the database the set belongs to
   * @param type - the id of the set type, something like 8xxx
   */
  PDBCatalogSet(const std::string &database, const std::string &name, const std::string &type, size_t setSize, PDBCatalogSetContainerType containerType) :
                setIdentifier(database + ":" + name),
                name(name),
                database(database),
                type(std::make_shared<std::string>(type)),
                setSize(setSize),
                containerType(containerType) {}

  /**
   * The set is a string of the form "dbName:setName"
   */
  std::string setIdentifier;

  /**
   * The name of the set
   */
  std::string name;

  /**
   * The database of the set
   */
  std::string database;

  /**
   * The size of the set
   */
  size_t setSize = 0;

  /**
   * The type of the set
   */
  std::shared_ptr<std::string> type;

  /**
   * The type of the container this set stores
   */
   int containerType = PDB_CATALOG_SET_NO_CONTAINER;

  /**
   * Return the schema of the database object
   * @return the schema
   */
  static auto getSchema() {

    // return the schema
    return sqlite_orm::make_table("sets",  sqlite_orm::make_column("setIdentifier", &PDBCatalogSet::setIdentifier),
                                           sqlite_orm::make_column("setName", &PDBCatalogSet::name),
                                           sqlite_orm::make_column("setDatabase", &PDBCatalogSet::database),
                                           sqlite_orm::make_column("setSize", &PDBCatalogSet::setSize),
                                           sqlite_orm::make_column("setType", &PDBCatalogSet::type),
                                           sqlite_orm::make_column("setContainerType", &PDBCatalogSet::containerType),
                                           sqlite_orm::foreign_key(&PDBCatalogSet::database).references(&PDBCatalogDatabase::name),
                                           sqlite_orm::foreign_key(&PDBCatalogSet::type).references(&PDBCatalogType::name),
                                           sqlite_orm::primary_key(&PDBCatalogSet::setIdentifier));
  }

};

}

#endif //PDB_PDBCATALOGSET_H
