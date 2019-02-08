

/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#ifndef PAGE_C
#define PAGE_C

#include <PDBPage.h>

#include "PDBBufferManagerImpl.h"
#include "PDBPage.h"
#include "PDBSet.h"

namespace pdb {

PDBPage :: PDBPage (PDBBufferManagerInterface &parent) : parent (parent), status(PDB_PAGE_NOT_LOADED), dirty(false) {}

void PDBPage :: incRefCount () {

	// lock the reference count
	unique_lock<mutex> blockLck(lk);

	// decrement the reference count
	refCount++;
}

void PDBPage :: decRefCount () {

	// lock the reference count
	unique_lock<mutex> blockLck(lk);

	// decrement the reference count
	refCount--;

	// grab a shared pointer
	auto spMe = me.lock();

	// did the reference count fall to zero
	if (refCount == 0) {

		// unlock the reference count
		blockLck.unlock();

		// if this is an anonymous page free it
		if (isAnon) {
			parent.freeAnonymousPage (spMe);
		}
		// ok the references of a page are down to zero deal with it.
		else {
			parent.downToZeroReferences (spMe);
		}
	}
}

size_t PDBPage :: whichPage () {
	return pageNum;
}

void PDBPage :: freezeSize (size_t numBytes) {
	auto spMe = me.lock();
	parent.freezeSize (spMe, numBytes);
}

bool PDBPage :: isPinned () {
	return pinned;
}

bool PDBPage :: isDirty () {
	return dirty;
}

void *PDBPage :: getBytes () {
	return bytes;
}

PDBSetPtr PDBPage :: getSet () {
	return whichSet;
}

void PDBPage :: setMe (PDBPagePtr toMe) {
	me = toMe;
}

void PDBPage :: unpin () {
	auto spMe = me.lock();
	parent.unpin (spMe);
}

void PDBPage :: repin () {
	auto spMe = me.lock();
	parent.repin (spMe);
}

void PDBPage :: setSet (PDBSetPtr inPtr) {
	whichSet = std::move(inPtr);
}

unsigned PDBPage :: numRefs () {
	return refCount;
}

PDBPageInfo &PDBPage :: getLocation () {
	return location;
}

PDBPageStatus &PDBPage::getStatus() {
  return status;
}

void PDBPage :: setPageNum (size_t inNum) {
	pageNum = inNum;
}

bool PDBPage :: isAnonymous () {
	return isAnon;
}

void PDBPage :: setAnonymous (bool arg) {
	isAnon = arg;
}

void PDBPage :: setBytes (void *locIn) {
	bytes = locIn;
}

bool PDBPage :: sizeIsFrozen () {
	return sizeFrozen;
}

void PDBPage :: setPinned () {
	pinned = true;
}

void PDBPage :: freezeSize () {
	sizeFrozen = true;
}

void PDBPage :: setUnpinned () {
	pinned = false;
}

void PDBPage :: setDirty () {
	dirty = true;
}

void PDBPage :: setClean () {
	dirty = false;
}

}

#endif

