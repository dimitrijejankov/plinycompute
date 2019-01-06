/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#ifndef PAGE_BACKEND_H
#define PAGE_BACKEND_H

#include "PDBPage.h"

namespace pdb {

class PDBPageBackend : public PDBPage {
 public:

  explicit PDBPageBackend(PDBStorageManagerInterface &manager) : PDBPage(manager) {};
};

}



#endif

