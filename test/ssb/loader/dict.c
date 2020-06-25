
/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include "../include/common.h"

#define	HHSIZE	(1024*1024)

/*
 * @file dict.c
 * Compress lineorder columns used in the Star Schema Benchmark using dictionary encoding.
 */

int main(int argc, char ** argv){

	int res = 0;

	if(argc !=3 ){
		printf("Usage: dictCompression inputColumn outputColumn\n");
		exit(-1);
	}

	struct columnHeader header;
	struct dictHeader dHeader;

	int inFd, outFd;
	long size, tupleNum;
	int numOfDistinct = 0;

	inFd = open(argv[1],O_RDONLY);

	if(inFd == -1){
		printf("Failed to open the input column\n");
		exit(-1);
	}

	read(inFd, &header, sizeof(struct columnHeader));

	if(header.format != UNCOMPRESSED){
		printf("The column has already been compressed. Nested Compression not supported yet\n");
		exit(-1);
	}


	long tupleOffset = 0;
	int blockTotal = header.blockTotal;
	long offset = 0;

	for(int j=0;j<blockTotal;j++){
		offset = j* sizeof(struct columnHeader) + tupleOffset * sizeof(int);
		lseek(inFd,offset,SEEK_SET);
		read(inFd,&header, sizeof(struct columnHeader));
		header.format = DICT;
		offset += sizeof(struct columnHeader);

		tupleNum = header.tupleNum;
		size = tupleNum * sizeof(int);
		char * content = (char *) malloc(size);

		if(!content){
			printf("Failed to allocate memory to accomodate the column\n");
			exit(-1);
		}

		char *table = (char *) mmap(0,size,PROT_READ,MAP_SHARED,inFd,offset);
		memcpy(content, table, size);
		munmap(table,size);
		close(inFd);

		int hashTable[HHSIZE] ;
		memset(hashTable,-1,sizeof(int) * HHSIZE);

		for(int i=0;i<tupleNum;i++){

			int key = ((int *)content)[i];
			int hKey = key % HHSIZE;

			if(hashTable[hKey] == -1){
				numOfDistinct ++;
				hashTable[hKey] = key;
			}else{
				if(hashTable[hKey] == key)
					continue;

				int j = 1;
				while(hashTable[hKey] != -1 && hashTable[hKey] != key){
					hKey = key % (HHSIZE + j*111) % HHSIZE; 
					j = j+1;
				}

				if(hashTable[hKey] == -1){
					hashTable[hKey] = key;
					numOfDistinct ++;
				}
			}
		}

		if(numOfDistinct > MAX_DICT_NUM)
			goto END;

		int numOfBits =1 ;

		while((1 << numOfBits) < numOfDistinct){
			numOfBits ++;
		}

		while(numOfBits % 8 !=0)
			numOfBits ++;

		if(numOfBits >= sizeof(int) * 8)
			goto END;

		int stride = sizeof(int) * 8 / numOfBits;

		if(stride <= 1)
			goto END;

		dHeader.dictNum = numOfDistinct;
		dHeader.bitNum = numOfBits;

		int * result = (int *) malloc(sizeof(int) * numOfDistinct);
		if(!result){
			printf("failed to allocate memory for result hash\n");
			exit(-1);
		}

		memset(result, -1, sizeof(int) * numOfDistinct);

		for(int i=0; i<HHSIZE;i++){
			if(hashTable[i] == -1)
				continue;

			int key = hashTable[i];
			int hKey = key % numOfDistinct;
			if( result[hKey] == -1){
				result[hKey] = key;
			}else{
				int j = 1;
				while(result[hKey] !=-1){
					hKey = key % (numOfDistinct + 111*j) % numOfDistinct;
					j++;
				}
				result[hKey] = key;
			}
		}

		for(int i=0;i<numOfDistinct;i++){
			dHeader.hash[i] = result[i];
		}


		int bitInInt = sizeof(int) * 8/ stride;

		outFd = open(argv[2],O_RDWR|O_CREAT);
		if(outFd == -1){
			printf("Failed to create output column\n");
			exit(-1);
		}

		long outOffset = 0;

		int * tmp = (int *) malloc(sizeof(int) * tupleNum);

		for(int i=0; i<tupleNum; i+= stride){

			int outInt = 0;

			for(int k=0;k<stride;k++){
				if((i+k) >= tupleNum)
					break;

				int key = ((int *)content)[k+i];
				int hKey = key % numOfDistinct;

				int j = 1;
				while(result[hKey] != key){
					hKey = key % (numOfDistinct + 111 * j) % numOfDistinct;
					j++;
				}

				hKey = hKey & 0xFFFF;
				memcpy((char *)(&outInt) + k*dHeader.bitNum/8, &hKey, dHeader.bitNum/8);

			}

			tmp[outOffset] = outInt;
			outOffset += 1; 

		}

		long blockSize = (4095 + sizeof(struct dictHeader) + outOffset * sizeof(int)) /4096 * 4096;
		header.blockSize = blockSize;

		write(outFd, &header, sizeof(struct columnHeader));
		write(outFd, &dHeader, sizeof(struct dictHeader));
		write(outFd,tmp,outOffset*sizeof(int));

		char buf[4096];
		memset(buf,0,sizeof(buf));
		int paddingSize = blockSize - sizeof(struct dictHeader) - outOffset * sizeof(int); 
		write(outFd, buf, paddingSize);

		close(outFd);
		free(content);
		free(tmp);
		free(result);

		tupleOffset += tupleNum;
	}

END:

	return res;
}
