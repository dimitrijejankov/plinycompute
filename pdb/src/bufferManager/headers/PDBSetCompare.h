

/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#ifndef SET_COMP_H
#define SET_COMP_H

#include "PDBSet.h"

namespace pdb {

// so that pages can be put into a map
struct PDBSetCompare {

 public:

  bool operator() (const PDBSetPtr &lhs, const PDBSetPtr &rhs) const {

	  // deal with the null case
	  if (lhs == nullptr && rhs != nullptr) {
		  return true;
	  } else if (rhs == nullptr) {
		  return false;
	  }

	  // otherwise, just compare the strings
	  if (lhs->getSetName () != rhs->getSetName ())
		  return lhs->getSetName () < rhs->getSetName ();

	  return lhs->getDBName () < rhs->getDBName ();
  }
};

}

#endif

