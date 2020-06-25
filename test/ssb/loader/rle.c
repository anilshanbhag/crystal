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
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "../include/common.h"

/*
 * @file rle.c
 * Compress a sorted foreign key column in LINEORDER table using Run Length encoding.
 *
 * Input:
 * 	@inputColumn: the column to be compressed using RLE.
 * 	@outputColumn: the name of the compressed column.
 */

int main(int argc, char ** argv){

	if(argc != 3){
		printf("./rleCompresssion inputColumn outputColumn\n");
		exit(-1);
	}

	int inFd = open(argv[1],O_RDONLY);
	if(inFd == -1){
		printf("Failed to open input file\n");
		exit(-1);
	}

	int outFd = open(argv[2],O_RDWR|O_CREAT);
	if(outFd == -1){
		printf("Failed to create output column\n");
		exit(-1);
	}

	struct columnHeader header;
	read(inFd, &header, sizeof(struct columnHeader));

	int blockTotal = header.blockTotal;

	long tupleOffset = 0;
	long offset = 0;

	for(int i=0;i<blockTotal;i++){
		offset = i*sizeof(struct columnHeader) + tupleOffset * sizeof(int);
		lseek(inFd,offset,SEEK_SET);
		read(inFd, &header, sizeof(struct columnHeader));
		offset += sizeof(struct columnHeader);
		long tupleNum = header.tupleNum;
		long size = tupleNum * sizeof(int);
        	char *content = (char *) malloc(size);
        	char *table =(char *) mmap(0,size,PROT_READ,MAP_SHARED,inFd,offset);
        	memcpy(content,table,size);
        	munmap(table,size);
        	close(inFd);

		tupleOffset += tupleNum;

		header.blockId = i;
		header.format = RLE;

		struct rleHeader rheader;

		int distinct = 1;
		int prev = ((int *)content)[0], curr;

		for(long i=1;i<tupleNum;i++){
			curr = ((int *)content)[i];
			if(curr == prev){
				continue;
			}
			distinct ++;
			prev = curr;
		}

		rheader.dictNum = distinct;
		int * dictValue = (int *)malloc(sizeof(int) * distinct);
		int * dictCount = (int *)malloc(sizeof(int) * distinct);
		int * dictPos = (int *)malloc(sizeof(int) * distinct);
		if(!dictPos || !dictCount || !dictValue){
			printf("Failed to allocate memory\n");
			exit(-1);
		}

		prev = ((int *)content)[0];
		int count = 1;
		int pos = 0;
		int k=0;
		for(long i =1; i<tupleNum; i++){
			curr = ((int *)content)[i];
			if(curr == prev){
				count ++;
				continue;
			}
			dictValue[k] = prev;
			dictPos[k] = pos;
			dictCount[k] = count;
			pos += count;
			k++;
			prev = curr;
			count = 1;
		}
		dictValue[k] = prev;
		dictPos[k] = pos;
		dictCount[k] = count;

		long blockSize = (4095+sizeof(struct rleHeader) + sizeof(int) * 3 *distinct)/4096 * 4096;
		char padding[4096];
		int padSize = blockSize - sizeof(struct rleHeader) - sizeof(int) * 3* distinct;

		memset(padding,0,sizeof(padding));
		header.blockSize = blockSize;

		write(outFd, &header, sizeof(struct columnHeader));
		write(outFd, &rheader, sizeof(struct rleHeader));
		write(outFd, dictValue, sizeof(int)*distinct);
		write(outFd, dictCount, sizeof(int)*distinct);
		write(outFd, dictPos, sizeof(int)*distinct);
		write(outFd,padding,padSize);

		close(outFd);
		free(content);
		free(dictValue);
		free(dictPos);
		free(dictCount);

	}

	return 0;

}
