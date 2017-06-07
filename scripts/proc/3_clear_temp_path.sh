#  Copyright 2018 Rice University                                           
#                                                                           
#  Licensed under the Apache License, Version 2.0 (the "License");          
#  you may not use this file except in compliance with the License.         
#  You may obtain a copy of the License at                                  
#                                                                           
#      http://www.apache.org/licenses/LICENSE-2.0                           
#                                                                           
#  Unless required by applicable law or agreed to in writing, software      
#  distributed under the License is distributed on an "AS IS" BASIS,        
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#  See the License for the specific language governing permissions and      
#  limitations under the License.                                           
#  ======================================================================== 
#!/bin/bash
# by Jia

pem_file=$1
user=$2

ip_len_valid=3
echo "-------------step2: clean PDB temp directory"

arr=($(awk '{print $0}' $PDB_HOME/conf/serverlist))
length=${#arr[@]}
for (( i=0 ; i<=$length ; i++ ))
do
        ip_addr=${arr[i]}
	if [ ${#ip_addr} -gt "$ip_len_valid" ]
	then
		echo -e "\n+++++++++++ clear PDB temp directory for server: $ip_addr +++++++++++"
            ssh -i $pem_file $user@$ip_addr 'rm -r -f ~/pdb_temp'
	fi
done



