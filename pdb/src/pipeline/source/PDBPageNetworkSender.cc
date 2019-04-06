#include <utility>

#include <utility>

//
// Created by dimitrije on 4/5/19.
//

#include <PDBPageNetworkSender.h>

#include "PDBPageNetworkSender.h"

pdb::PDBPageNetworkSender::PDBPageNetworkSender(string address, int32_t port, const std::pair<uint64_t, std::string> &pageSetID, pdb::PDBPageQueuePtr queue)
    : address(std::move(address)), port(port), queue(std::move(queue)) {}

bool pdb::PDBPageNetworkSender::setup() {
  return false;
}

bool pdb::PDBPageNetworkSender::run() {
  return false;
}
