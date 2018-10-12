

#ifndef BE_STORAGE_MGR_H
#define BE_STORAGE_MGR_H

#include <PDBCommunicator.h>
#include <PDBServer.h>
#include "PDBStorageManagerInterface.h"


namespace pdb {

/**
 * This is the part of the storage manager that is running in the back end.
 * There are two storage managers running on each machine: one on the front
 * end and one on the back end.
 *
 * Note that all communication via the front end and the back end storage managers
 * happens using the ServerFunctionality interface that both implement (though
 * the only time that the front end storage manager directly contacts the back end
 * is on startup, to send any necessary initialization information).  The ONE SUPER
 * IMPORTANT exception to this is that NO DATA on any page is ever sent over a
 * PDBCommunicator object.  Rather, when a page is requested by the back end or
 * a page is sent from the front end to the back end, the page is allocated into
 * the buffer pool, which is memory shared by the back end and the front end.  What
 * is actually sent over the PDBCommunicator is only a pointer into the shared
 * memory.
 *
 * The end doesn't do much work.  All it does is (perhaps?) some internal bookkepping
 * to manage the pages it has received from the front end.  It does not actually buffer
 * anything; all of the buffering happens in the front end.  It basically just forwards
 * requests from the pages to the front end.
 */
class PDBStorageManagerBackEnd : public PDBStorageManagerInterface {

public:

  void registerHandlers(PDBServer &forMe) override;

};

}



#endif


