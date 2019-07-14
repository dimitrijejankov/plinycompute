#include "BufManagerRequestBase.h"

namespace pdb {

// init the timestamp
std::atomic<std::uint64_t> BufManagerRequestBase::lastID;

}
