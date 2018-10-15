/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#ifndef PDB_PAGEDREQUEST_CC
#define PDB_PAGEDREQUEST_CC

namespace pdb {

/**
 * This templated function makes it easy to write a simple network client that asks a request,
 * then gets a result.
 * @tparam RequestType - this is the type of the object we are sending
 * @tparam ResponseType - the type of the response we are expecting from the other side
 * @tparam ReturnType - the return type
 * @tparam RequestTypeParams - the parameters of the request
 * @param myLogger - The logger we write error messages to
 * @param port - the port to send the request to
 * @param address - the address to send the request to
 * @param onErr - the value to return if there is an error sending/receiving data
 * @param bytesForRequest - the number of bytes to give to the allocator used to build the request
 * @param processResponse - the function used to process the response to the request
 * @param args - the arguments to give to the constructor of the request
 * @return whatever the return ends up being
 */
template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType pagedRequest(PDBLoggerPtr myLogger,
                        int port,
                        std::string address,
                        ReturnType onErr,
                        size_t bytesForRequest,
                        function<ReturnType(Handle<ResponseType>)> processResponse,
                        RequestTypeParams&&... args);

}

#endif //PDB_PAGEDREQUEST_H
