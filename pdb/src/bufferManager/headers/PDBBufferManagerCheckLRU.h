
/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#include <utility> 

#ifndef CHECK_LRU_H
#define CHECK_LRU_H

using namespace std;

namespace pdb {

// so that pages can be searched based on LRU access time
class PDBBufferManagerCheckLRU {

 public:
  bool operator() (const pair <void *, size_t> lhs, const pair <void *, size_t> rhs) const {
	  return lhs.second < rhs.second;
  }
};

}



#endif

