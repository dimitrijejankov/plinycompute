#include "BufManagerRequestBase.h"


#ifdef DEBUG_BUFFER_MANAGER

namespace pdb {

// init the timestamp
std::atomic<std::uint64_t> BufManagerRequestBase::lastID;

}

#endif