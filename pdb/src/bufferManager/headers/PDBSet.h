

/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#ifndef PC_SET_H
#define PC_SET_H

#include <memory>
#include <string>

namespace pdb {

// create a smart pointer for pages
using namespace std;
class PDBSet;
typedef shared_ptr <PDBSet> PDBSetPtr;

class PDBSet {

 public:

  // the name of the set
  string getSetName () {
	  return setName;
  }

  // the name of the database that the set is part of
  string getDBName () {
	  return dbName;
  }

  // create a set with the given name, database, physical storage location
  PDBSet (string dbNameIn, string setNameIn) {
	  setName = std::move(setNameIn);
	  dbName = std::move(dbNameIn);
  }

  bool operator==(const PDBSet &rhs) const {
    return setName == rhs.setName &&
        dbName == rhs.dbName;
  }

  bool operator!=(const PDBSet &rhs) const {
    return !(rhs == *this);
  }

 private:

  string setName;
  string dbName;

};

}


#endif

