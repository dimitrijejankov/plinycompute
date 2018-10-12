

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

#include "PDBStorageManager.h"
#include "PDBPage.h"
#include "PDBSet.h"

namespace pdb {

PDBPage :: PDBPage (PDBStorageManager &parent) : parent (parent) {}

void PDBPage :: incRefCount () {
	refCount++;
}

void PDBPage :: decRefCount () {
	refCount--;
	auto spMe = me.lock();
	if (refCount == 0) {
		if (isAnon)
			parent.freeAnonymousPage (spMe);
		else 
			parent.downToZeroReferences (spMe);
		
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

