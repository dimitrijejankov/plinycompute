#pragma once
#include <string>
#include <sqlite_orm.h>

namespace pdb {

/**
 * This is just a definition for the shared pointer on the type
 */
class PDBCatalogSetOnNode;
typedef std::shared_ptr<PDBCatalogSetOnNode> PDBCatalogSetOnNodePtr;

class PDBCatalogSetOnNode {

 public:

  /**
   * The default constructor needed by the orm
   */
  PDBCatalogSetOnNode() = default;

  /**
   * The initializer constructor
   * @param setIdentifier - the identifier of the set
   * @param nodeID - the identifier of the node
   */
  PDBCatalogSetOnNode(std::string setIdentifier, int32_t nodeID) : setIdentifier(std::move(setIdentifier)),
                                                                                 nodeID(nodeID) {}

  /**
   * The initializer constructor
   * @param setIdentifier - the identifier of the set
   * @param nodeID - the identifier of the node
   * @param recordCount - how many records are in this shard
   * @param shardSize - how many bytes are there in this shard
   */
  PDBCatalogSetOnNode(std::string setIdentifier,
                      int32_t nodeID,
                      uint64_t recordCount,
                      uint64_t shardSize,
                      uint64_t keyCount,
                      uint64_t keySize) : setIdentifier(std::move(setIdentifier)),
                                          nodeID(nodeID),
                                          recordCount(recordCount),
                                          shardSize(shardSize),
                                          keyCount(keyCount),
                                          keySize(keySize) {}

  /**
   * The identifier of the set
   */
  std::string setIdentifier;

  /**
   * The identifier of the node
   */
  int32_t nodeID{};

  /**
   * The record count
   */
  uint64_t recordCount = 0;

  /**
   * The size of the shard on this node in bytes
   */
  uint64_t shardSize = 0;

  /**
   * The key count on this node, a node always holds all the keys
   */
  uint64_t keyCount = 0;

  /**
   * The size of the keys on this node, a node always holds all the keys
   */
  uint64_t keySize = 0;

  /**
   * Return the schema of the database object
   * @return the schema
   */
  static auto getSchema() {

    // return the schema
    return sqlite_orm::make_table("setOnNode",
                                  sqlite_orm::make_column("setIdentifier", &PDBCatalogSetOnNode::setIdentifier),
                                  sqlite_orm::make_column("nodeID", &PDBCatalogSetOnNode::nodeID),
                                  sqlite_orm::make_column("recordCount", &PDBCatalogSetOnNode::recordCount),
                                  sqlite_orm::make_column("shardSize", &PDBCatalogSetOnNode::shardSize),
                                  sqlite_orm::make_column("keyCount", &PDBCatalogSetOnNode::keyCount),
                                  sqlite_orm::make_column("keySize", &PDBCatalogSetOnNode::keySize),
                                  sqlite_orm::primary_key(&PDBCatalogSetOnNode::setIdentifier, &PDBCatalogSetOnNode::nodeID));
  }
};

}