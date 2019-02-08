
#ifndef CATALOG_C
#define CATALOG_C

#include "PDBBufferManagerFileWriter.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace pdb {

void PDBBufferManagerFileWriter :: putString (string key, string value) {
	myData [key] = std::move(value);
}

void PDBBufferManagerFileWriter :: putStringList (string key, vector <string> value) {
	string res;
	for (const string &s : value) {
		res += s + "#";
	}
	myData [key] = res;
}

void PDBBufferManagerFileWriter :: putUnsignedLong(string key, uint64_t value) {
	ostringstream convert;
	convert << value;
	myData [key] = convert.str ();
}

void PDBBufferManagerFileWriter :: putLong(string key, int64_t value) {
	ostringstream convert;
	convert << value;
	myData [key] = convert.str ();
}

bool PDBBufferManagerFileWriter :: getStringList (string key, vector <string> &returnVal) {

	// verify the entry is in the map
	if (myData.count (key) == 0)
		return false;

	// it is, so parse the other side
	string res = myData[key];	
	for (size_t pos = 0; pos < (int) res.size (); pos = res.find ('#', pos + 1) + 1) {
		string temp = res.substr (pos, res.find ('#', pos + 1) - pos);
		returnVal.push_back (temp);
	}
	return true;
}

bool PDBBufferManagerFileWriter :: getString (string key, string &res) {
	if (myData.count (key) == 0)
		return false;

	res = myData[key];
	return true;
}

bool PDBBufferManagerFileWriter :: getUnsignedLong(string key, uint64_t &value) {

	// verify the entry is in the map
	if (myData.count (key) == 0)
		return false;

	// it is, so convert it to an int
	string :: size_type sz;
	try {
		value = std::stoul (myData [key], &sz);

	// exception means that we could not convert
	} catch (...) {
		return false;
	}
	
	return true;
}

bool PDBBufferManagerFileWriter :: getLong(string key, int64_t &value) {

	// verify the entry is in the map
	if (myData.count (key) == 0)
		return false;

	// it is, so convert it to an int
	string :: size_type sz;
	try {
		value = std::stol (myData [key], &sz);

		// exception means that we could not convert
	} catch (...) {
		return false;
	}

	return true;
}


PDBBufferManagerFileWriter :: PDBBufferManagerFileWriter (string fNameIn) {

	// remember the catalog name
	fName = std::move(fNameIn);

	// try to open the file
	string line;
	ifstream myfile (fName);

	// if we opened it, read the contents
	if (myfile.is_open()) {

		// loop through all of the lines
    		while (getline (myfile,line)) {

			// find how to cut apart the string
			size_t firstPipe, secPipe, lastPipe;
			firstPipe = line.find ('|');
			secPipe = line.find ('|', firstPipe + 1);
			lastPipe = line.find ('|', secPipe + 1);

			// if there is an error, don't add anything
			if (firstPipe >= (int) line.size () || secPipe >= (int) line.size () || lastPipe >= (int) line.size ())
				continue;

			// and add the pair
			myData [line.substr (firstPipe + 1, secPipe - firstPipe - 1)] = 
				line.substr (secPipe + 1, lastPipe - secPipe - 1);
		}
		myfile.close();
	}
}

void PDBBufferManagerFileWriter :: save () {

	ofstream myFile (fName, ofstream::out | ofstream::trunc);
	if (myFile.is_open()) {
		for (auto const &ent : myData) {
			myFile << "|" << ent.first << "|" << ent.second << "|\n";
		}
	}
}

}

#endif


