//
// Created by dimitrije on 4/5/19.
//

#include <PDBPageNetworkSender.h>

#include "PDBPageNetworkSender.h"

pdb::PDBPageNetworkSender::PDBPageNetworkSender(const string &address, int32_t port, const pdb::PDBPageQueuePtr &queue)
    : address(address), port(port), queue(queue) {}

bool pdb::PDBPageNetworkSender::setup() {
  return false;
}

bool pdb::PDBPageNetworkSender::run() {
  return false;
}
